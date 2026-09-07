"""Live, styled TUI for the benchmark, built on `rich` (Evil Wizard theme).

Three phases:
  1. LANDING  - the Archmage of Benchmarks, arms raised, channeling power over a
                shifting field of magical runes, then auto-start.
  2. RUNNING  - a battle scene: the wizard on the LEFT hurls spells/lightning at
                the current model (named) on the RIGHT, beside live progress
                bars, the leaderboard, and a recent-case log.
  3. RESULTS  - a big GAME OVER screen with the full results table printed in the
                TUI and a surprise finale.

The benchmark itself runs on a background thread; the engine drives all state
through its event callback, so this file is purely presentation.
"""

import json
import os
import random
import threading
import time
from collections import deque

from rich.align import Align
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, TextColumn,
                           TimeElapsedColumn, MofNCompleteColumn)
from rich.table import Table
from rich.text import Text

from harness import art, engine, report


# ------------------------------------------------------------------ helpers

def _fmt_size(num_bytes: int) -> str:
    mb = num_bytes / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"


def select_models_menu(config_path: str):
    """Interactive pre-run menu: list models with prior performance and let the
    user choose what to run (all / score < X / score > X).

    Returns a list of model file paths to run, or None for "all". Falls back to
    None (all) when stdin is not interactive.
    """
    import sys
    from rich.console import Console
    from harness import engine, history

    console = Console()
    cfg = engine.load_yaml(config_path)
    all_paths = engine.find_models(cfg["models_dir"], cfg.get("exclude_dirs"))
    if not all_paths:
        return None  # engine will raise a clear error later

    scores = history.latest_scores()

    # Show the roster with last-known performance.
    table = Table(title="[bold magenta]THE ARCHMAGE'S ROSTER",
                  header_style="bold magenta", border_style="magenta")
    table.add_column("Model", overflow="fold")
    table.add_column("Size", justify="right")
    table.add_column("Last Score", justify="right")
    table.add_column("Last Run")
    for path in all_paths:
        name = os.path.basename(path)
        rec = scores.get(name)
        if rec:
            score_txt = Text(f"{rec['score']}/{rec['total']}",
                             style="green" if rec["status"] == "ok" else "red")
            when = rec["ts"]
        else:
            score_txt = Text("never run", style="dim")
            when = "-"
        table.add_row(name, _fmt_size(os.path.getsize(path)), score_txt, when)
    console.print(table)

    if not sys.stdin.isatty():
        console.print("[dim]non-interactive stdin: running all models.[/dim]")
        return None

    console.print(
        "\n[bold]Choose what to run:[/bold]\n"
        "  [bold magenta]A[/bold magenta]) all models   "
        "[bold magenta]N[/bold magenta]) untested only   "
        "[bold magenta]<[/bold magenta]) score below X   "
        "[bold magenta]>[/bold magenta]) score above X   "
        "([dim]Enter = all[/dim])"
    )
    choice = input("selection > ").strip().lower()

    if choice in ("", "a", "all"):
        return None

    if choice in ("n", "new", "untested"):
        selected, info = history.filter_models(all_paths, "untested")
        if not selected:
            console.print("[yellow]no untested models; everything has been run.[/yellow]")
            return None
        console.print(f"[green]{len(selected)} untested model(s) selected:[/green] "
                      + ", ".join(os.path.basename(p) for p in selected))
        return selected

    if choice in ("<", ">"):
        raw = input(f"score threshold X (total, e.g. 12) > ").strip()
        try:
            threshold = int(raw)
        except ValueError:
            console.print("[red]invalid number; running all.[/red]")
            return None
        mode = "lt" if choice == "<" else "gt"
        selected, info = history.filter_models(all_paths, mode, threshold)
        if not selected:
            console.print("[yellow]no models match that filter; running all.[/yellow]")
            return None
        console.print(f"[green]{len(selected)} model(s) selected:[/green] "
                      + ", ".join(os.path.basename(p) for p in selected))
        return selected

    console.print("[yellow]unrecognized choice; running all.[/yellow]")
    return None


def _red_art(lines, style="bold red") -> Text:
    t = Text()
    for i, ln in enumerate(lines):
        t.append(ln + "\n", style=style)
    return t


# ------------------------------------------------------------------ state

class DashboardState:
    """Shared state mutated by the worker thread, read by the render loop."""

    def __init__(self):
        self.lock = threading.Lock()
        self.run_id = ""
        self.out_dir = ""
        self.start_time = time.time()
        self.model_count = 0
        self.case_count = 0
        self.test_ids = []

        self.models = {}
        self.model_order = []

        self.current_model = None
        self.current_model_index = 0
        self.current_case_done = 0
        self.current_test = None
        self.current_case = None

        self.log = deque(maxlen=10)
        self.finished = False
        self.results = None

    def handle(self, event: str, payload: dict) -> None:
        with self.lock:
            getattr(self, f"_on_{event}", lambda p: None)(payload)

    def _on_run_start(self, p):
        self.run_id = p["run_id"]
        self.out_dir = p["out_dir"]
        self.model_count = p["model_count"]
        self.case_count = p["case_count"]
        self.test_ids = p["test_ids"]

    def _on_model_start(self, p):
        name = p["model_name"]
        self.current_model = name
        self.current_model_index = p["index"]
        self.current_case_done = 0
        if name not in self.models:
            self.model_order.append(name)
        self.models[name] = {
            "size": p["size_bytes"], "status": "loading",
            "passed": 0, "total": 0, "tps": [], "load_time_s": 0.0,
            "fails": [],   # list of (test_id, case_id, reason) for failed cases
        }

    def _on_model_loaded(self, p):
        m = self.models.get(p["model_name"])
        if m:
            m["status"] = "running"
            m["load_time_s"] = p["load_time_s"]

    def _on_test_start(self, p):
        self.current_test = p["test_id"]

    def _on_model_error(self, p):
        m = self.models.get(p["model_name"])
        if m:
            m["status"] = "error"
        self.log.appendleft(("ERR", p["model_name"], "-", str(p.get("error", ""))[:40], 0))

    def _on_case_done(self, p):
        m = self.models.get(p["model_name"])
        tps = p["stats"].get("tokens_per_s", 0)
        if m:
            m["total"] += 1
            if p["passed"]:
                m["passed"] += 1
            else:
                m["fails"].append((p["test_id"], p["case_id"], p.get("reason", "")))
            if tps:
                m["tps"].append(tps)
        self.current_case_done += 1
        self.current_test = p["test_id"]
        self.current_case = p["case_id"]
        self.log.appendleft((
            "PASS" if p["passed"] else "FAIL",
            p["model_name"], p["test_id"], p["case_id"], tps,
        ))

    def _on_model_done(self, p):
        m = self.models.get(p["model_name"])
        if m and m["status"] != "error":
            m["status"] = "done"

    def _on_run_done(self, p):
        self.finished = True
        self.results = p["results"]

    # --- aggregate failure helpers (call with self.lock held) ---

    def failure_summary(self):
        """Return (models_with_failures, total_failed_cases, error_models)."""
        models_with_fails = 0
        total_failed = 0
        error_models = 0
        for name in self.model_order:
            m = self.models[name]
            if m["status"] == "error":
                error_models += 1
            n_fail = len(m["fails"])
            if n_fail:
                models_with_fails += 1
                total_failed += n_fail
        return models_with_fails, total_failed, error_models


# ------------------------------------------------------------------ renderables

def _wizard_over_runes(figure_lines, tick: int, width: int, height: int,
                       fig_style="bold magenta", align="center") -> Text:
    """Render an ASCII figure over a shifting field of magical runes."""
    runes = art.rune_block(width, height, density=0.06 + 0.03 * (tick % 3))
    canvas = [list(row.ljust(width)) for row in runes]

    top = max(0, (height - len(figure_lines)) // 2)
    for i, line in enumerate(figure_lines):
        row = top + i
        if row >= height:
            break
        if align == "center":
            left = max(0, (width - len(line)) // 2)
        else:  # left align with small margin
            left = 2
        for j, ch in enumerate(line):
            if left + j < width and ch != " ":
                canvas[row][left + j] = ch

    body = Text()
    for row in canvas:
        for ch in row:
            if ch in art._RUNE_GLYPHS and ch not in "()/\\|_-.":
                # flickering purple/blue magical motes behind the figure
                shade = random.choice(["magenta", "blue", "bright_black", "cyan"])
                body.append(ch, style=shade)
            elif ch == " ":
                body.append(" ")
            else:
                body.append(ch, style=fig_style)
        body.append("\n")
    return body


def _battle_scene(state: DashboardState, tick: int) -> Panel:
    """The wizard (left) casts a spell that IS the current test, flying toward
    the model (right). The spell wraps the running test name."""
    with state.lock:
        model = state.current_model or "(summoning target)"
        m = state.models.get(model, {}) if state.current_model else {}
        status = m.get("status", "")
        passed = m.get("passed", 0)
        total = m.get("total", 0)
        test_id = state.current_test
        case_id = state.current_case

    # Slow the animation down: advance the spell shimmer every 3 render ticks.
    slow = tick // 3

    wiz = art.wizard_battle_lines()

    # The spell carries the running test's name. Prefer a readable label.
    if test_id:
        spell = art.spell_for_test(test_id)
        label = {"test1_auth_gating": "AUTH-GATE",
                 "test2_string_manipulation": "STRING-HEX"}.get(test_id, test_id)
        spell_text = art.render_spell(test_id, label, slow)
        spell_style = spell["style"]
        spell_name = spell["label"]
    else:
        spell = None
        spell_text = "~ * ~ gathering mana ~ * ~"
        spell_style = "dim magenta"
        spell_name = "CHANNELING"

    body = Text()

    # Find the wizard's casting-hand row (the line containing '@').
    hand_row = next((i for i, ln in enumerate(wiz) if "@" in ln), len(wiz) // 2)

    # Target card (right side).
    target_card = [
        ".----------------------------.",
        "|  TARGET                    |",
        f"|  {model[:24]:<24}  |",
        f"|  status: {status:<16}  |",
        f"|  score:  {((str(passed)+'/'+str(total)) if total else '--'):<16}  |",
        "'----------------------------'",
    ]

    max_lines = max(len(wiz), len(target_card) + 3)
    wiz_w = max((len(l) for l in wiz), default=16)

    for i in range(max_lines):
        seg = Text()
        wline = wiz[i] if i < len(wiz) else ""
        # Trim the '@' marker off the wizard art; the spell renders separately.
        wline_clean = wline.replace("@", " ")
        seg.append(f"{wline_clean:<{wiz_w}}", style="bold magenta")

        # The spell flies from the hand row.
        if i == hand_row:
            seg.append("  ", style="default")
            seg.append(spell_text, style=spell_style)
        else:
            seg.append(" " * (len(spell_text) + 2))

        # Target card, vertically centered a bit lower than the hand.
        tline_idx = i - (hand_row - 2)
        if 0 <= tline_idx < len(target_card):
            hit = (slow % 2 == 0) and status in ("running", "loading")
            tstyle = "bold red" if hit else "white"
            # pad so the card starts at a consistent column
            seg.append("   " + target_card[tline_idx], style=tstyle)

        body.append_text(seg)
        body.append("\n")

    # Caption: which spell / test is being cast, steady (not frantic).
    body.append("\ncasting ", style="magenta")
    body.append(spell_name, style=spell_style)
    if case_id:
        body.append(f"   [ {case_id} ]", style="dim")
    if status in ("running", "loading") and slow % 2 == 0:
        body.append("   IMPACT!", style="bold yellow")

    return Panel(body, title="[bold magenta]:: THE ARCHMAGE CASTS ::",
                 border_style="magenta")


def _leaderboard(state: DashboardState) -> Table:
    table = Table(title="[bold magenta]GRIMOIRE OF TARGETS", expand=True,
                  header_style="bold magenta", border_style="magenta")
    table.add_column("", justify="center", width=3)   # failure marker
    table.add_column("Model", overflow="fold")
    table.add_column("Size", justify="right")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Fails", justify="right")
    table.add_column("tok/s", justify="right")

    status_style = {"loading": "yellow", "running": "cyan",
                    "done": "green", "error": "red", "queued": "dim"}
    for name in state.model_order:
        m = state.models[name]
        avg_tps = round(sum(m["tps"]) / len(m["tps"]), 1) if m["tps"] else 0
        score_txt = f"{m['passed']}/{m['total']}" if m["total"] else "-"
        n_fail = len(m["fails"])
        has_error = m["status"] == "error"

        if has_error:
            marker = Text("X", style="bold red")
            score_style = "red"
        elif n_fail:
            marker = Text("!", style="bold red")     # at least one failed case
            score_style = "red"
        elif m["total"] and m["status"] == "done":
            marker = Text("+", style="bold green")   # completed, all passed
            score_style = "bold green"
        else:
            marker = Text(".", style="dim")          # in progress / not started
            score_style = "white"

        fails_txt = Text(str(n_fail) if m["total"] else "-",
                         style="bold red" if n_fail else "dim")

        table.add_row(
            marker, name, _fmt_size(m["size"]),
            Text(m["status"].upper(), style=status_style.get(m["status"], "white")),
            Text(score_txt, style=score_style),
            fails_txt,
            str(avg_tps),
        )
    return table


def _log_panel(state: DashboardState) -> Panel:
    lines = []
    for status, model, test, case, tps in state.log:
        color = {"PASS": "green", "FAIL": "red", "ERR": "red bold"}.get(status, "white")
        tps_txt = f"{tps} tok/s" if tps else ""
        lines.append(Text.assemble(
            (f"[{status}] ", color), (f"{model} ", "bold"),
            (f"{test}/{case} ", "dim"), (tps_txt, "cyan"),
        ))
    body = Group(*lines) if lines else Text("the duel has not yet begun...", style="dim")
    return Panel(body, title="[bold magenta]SPELLS CAST", border_style="magenta")


# ------------------------------------------------------------------ phases

def _render_landing(tick: int) -> Group:
    # The archmage, arms raised, channeling power over a rune field that fills
    # the panel. Pulsing title + "power level" charge bar.
    wiz_lines = art.wizard_summon_lines()
    wizard = _wizard_over_runes(wiz_lines, tick, width=64, height=24,
                                fig_style="bold magenta", align="center")

    pulse = ["bold magenta", "bold bright_magenta", "bold red", "bold bright_magenta"][tick % 4]
    title = Text("T H E   A R C H M A G E   O F   B E N C H M A R K S", style=pulse)

    charge = min(tick, 20)
    bar = "[" + ("#" * charge).ljust(20, ".") + "]"
    spark = ["*  ", " * ", "  *", " * "][tick % 4]
    sub = Text.assemble(
        ("\ndark energies gather to test the small models...\n\n", "magenta"),
        (f"{spark} channeling power {bar}\n", "bold bright_cyan"),
        ("the duel begins shortly...", "dim magenta"),
    )
    inner = Group(Align.center(title), Align.center(wizard), Align.center(sub))
    return Group(Align.center(Panel(inner, border_style="magenta", padding=(1, 3),
                                    title="[bold magenta]:: SUMMONING CIRCLE ::")))


def _render_running(state: DashboardState, tick: int, overall, per_model,
                    overall_task, model_task) -> Group:
    with state.lock:
        elapsed = int(time.time() - state.start_time)
        done_models = sum(1 for n in state.model_order
                          if state.models[n]["status"] in ("done", "error"))
        overall.update(overall_task, total=max(state.model_count, 1),
                       completed=done_models)
        cpm = state.case_count or 1
        per_model.update(model_task, total=cpm,
                         completed=min(state.current_case_done, cpm),
                         description=f"[bold red]cases[/] ({state.current_model or '-'})")
        leaderboard = _leaderboard(state)
        log_panel = _log_panel(state)
        models_with_fails, total_failed, error_models = state.failure_summary()

    # Live failure indicator: green ALL CLEAR until the first failure, then a
    # blinking alert with running counts.
    if total_failed or error_models:
        alert = "[!]" if tick % 2 == 0 else "[ ]"
        fail_seg = (
            f"| {alert} FELLED: {total_failed} case(s) across "
            f"{models_with_fails} model(s)"
            + (f", {error_models} banished" if error_models else ""),
            "bold red blink",
        )
    else:
        fail_seg = ("| [OK] no model has fallen yet", "bold green")

    header = Text.assemble(
        ("ARCHMAGE'S TRIAL  ", "bold bright_magenta"),
        (f"run {state.run_id}  ", "magenta"),
        (f"| elapsed {elapsed}s  ", "cyan"),
        (f"| duel {state.current_model_index}/{state.model_count} ", "dim"),
        fail_seg,
    )
    right = Group(
        Panel(Group(overall, per_model), title="[bold magenta]RITUAL PROGRESS",
              border_style="magenta"),
        leaderboard,
        log_panel,
    )
    left = _battle_scene(state, tick)
    return Group(header, Columns([left, right], expand=True, equal=False))


def _render_results(state: DashboardState, tick: int = 0, wait_key: bool = False) -> Group:
    results = state.results or {}
    rows = report._model_summary_rows(results) if results.get("models") else []
    best = next((r for r in rows if r["all_pass"]), None)

    with state.lock:
        models_with_fails, total_failed, error_models = state.failure_summary()

    # --- big GAME OVER banner at the top, pulsing ---
    go_style = ["bold red", "bold bright_red", "bold magenta", "bold bright_red"][tick % 4]
    game_over = _red_art(art.game_over_lines(), style=go_style)

    # --- surprise finale: the wizard's verdict, framed by floating skulls ---
    # If a champion emerged, the wizard "spares" it; otherwise total annihilation.
    total_models = len(rows)
    fallen = models_with_fails + error_models
    if best:
        champion_line = Text.assemble(
            ("*** A CHAMPION SURVIVES THE ARCHMAGE ***\n", "bold bright_green"),
            (f"  {best['model']}  ", "bold bright_green"),
            (f"[{_fmt_size(best['size_bytes'])}, {best['avg_tps']} tok/s]\n", "cyan"),
            ("it alone bent the flag to the user's will and lived.", "green"),
        )
        finale_border = "green"
    else:
        # Pick the "least defeated" as the wizard's grudging favorite.
        toughest = rows[0] if rows else None
        champ_txt = (f"the sturdiest was {toughest['model']} ({toughest['score']})"
                     if toughest else "none stood a chance")
        champion_line = Text.assemble(
            ("*** THE ARCHMAGE REMAINS UNDEFEATED ***\n", "bold red"),
            (f"all {total_models} models fell to the trials.\n", "red"),
            (champ_txt, "yellow"),
        )
        finale_border = "red"

    # Animated skulls flanking the verdict.
    sway = "  " if tick % 2 == 0 else "   "
    skull_row = Text(f"{sway}(x_x)      >:)  cackle...      (x_x){sway}",
                     style="bold magenta")
    finale = Panel(
        Group(Align.center(champion_line), Text(""), Align.center(skull_row)),
        title="[bold magenta]:: THE ARCHMAGE'S VERDICT ::",
        border_style=finale_border,
    )

    # --- full results table, printed right here in the TUI ---
    table = Table(title="[bold magenta]FINAL GRIMOIRE", expand=True,
                  header_style="bold magenta", border_style="magenta")
    table.add_column("#", justify="right")
    table.add_column("Model", overflow="fold")
    table.add_column("Size", justify="right")
    table.add_column("Survived", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("tok/s", justify="right")
    for i, r in enumerate(rows, 1):
        table.add_row(
            str(i), r["model"], _fmt_size(r["size_bytes"]),
            Text("YES", style="bold green") if r["all_pass"] else Text("no", style="red"),
            Text(r["score"], style="bold green" if r["all_pass"] else "white"),
            str(r["avg_tps"]),
        )

    # --- tally line ---
    if total_failed or error_models:
        tally = Text.assemble(
            ("[!] CASUALTIES  ", "bold red"),
            (f"{total_failed} case(s) felled across {models_with_fails} model(s)", "red"),
            (f"  |  {error_models} banished (load errors)" if error_models else "", "red"),
        )
    else:
        tally = Text("[OK] FLAWLESS - every model survived every trial.", style="bold green")

    # --- detailed failure breakdown ---
    failure_panel = None
    with state.lock:
        fail_lines = []
        for name in state.model_order:
            m = state.models[name]
            if m["status"] == "error":
                fail_lines.append(Text.assemble(
                    ("X ", "bold red"), (f"{name}  ", "bold"),
                    ("banished - failed to load/run", "red"),
                ))
                continue
            if m["fails"]:
                fail_lines.append(Text.assemble(
                    ("! ", "bold red"), (f"{name}  ", "bold"),
                    (f"{len(m['fails'])} trial(s) failed:", "red"),
                ))
                for test_id, case_id, reason in m["fails"][:6]:
                    leak = "LEAK" in reason.upper() or "LEAKED" in reason.upper()
                    tag = "  -> [LEAK] " if leak else "  -> "
                    fail_lines.append(Text.assemble(
                        (tag, "bold red" if leak else "yellow"),
                        (f"{test_id}/{case_id}  ", "dim"),
                        (reason, "red" if leak else "yellow"),
                    ))
                extra = len(m["fails"]) - 6
                if extra > 0:
                    fail_lines.append(Text(f"    ... and {extra} more", style="dim"))
    if fail_lines:
        failure_panel = Panel(Group(*fail_lines),
                              title="[bold red]:: THE FALLEN ::",
                              border_style="red")

    parts = [
        Align.center(game_over),
        finale,
        Align.center(tally),
        Text(""),
        table,
    ]
    if failure_panel is not None:
        parts.append(failure_panel)
    parts.append(Text(f"\nscrolls archived at: {state.out_dir}", style="dim"))

    if wait_key:
        # Blinking prompt so the user knows the screen waits for them.
        if tick % 2 == 0:
            prompt = Text(">>> press any key to banish the Archmage and exit <<<",
                          style="bold bright_magenta")
        else:
            prompt = Text("    press any key to banish the Archmage and exit    ",
                          style="magenta")
        parts.append(Align.center(prompt))

    inner = Group(*parts)
    return Group(Panel(inner, border_style="magenta", padding=(1, 2),
                       title="[bold red]::: GAME OVER :::"))


# ------------------------------------------------------------------ driver

def _key_pressed() -> bool:
    """Non-blocking check for any keypress. Windows uses msvcrt; POSIX uses
    select on stdin. Returns True if a key was pressed (and consumes it)."""
    try:
        import msvcrt  # Windows
        if msvcrt.kbhit():
            msvcrt.getch()
            return True
        return False
    except ImportError:
        import select
        import sys as _sys
        dr, _, _ = select.select([_sys.stdin], [], [], 0)
        if dr:
            _sys.stdin.read(1)
            return True
        return False


def _drain_keys() -> None:
    """Consume any buffered keystrokes so a leftover Enter (e.g. from the
    pre-run menu) doesn't immediately dismiss the results screen."""
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        import select
        import sys as _sys
        while select.select([_sys.stdin], [], [], 0)[0]:
            _sys.stdin.read(1)


def run_tui(config_path: str, landing_seconds: float = 3.0,
            results_hold_seconds: float = 0.0, model_filter=None) -> None:
    state = DashboardState()

    overall = Progress(TextColumn("[bold magenta]duels"), BarColumn(complete_style="magenta"),
                       MofNCompleteColumn(), TimeElapsedColumn())
    per_model = Progress(TextColumn("[bold magenta]cases"), BarColumn(complete_style="bright_magenta"),
                         MofNCompleteColumn())
    overall_task = overall.add_task("models", total=1)
    model_task = per_model.add_task("cases", total=1)

    error_holder = {}
    started = threading.Event()

    def worker():
        started.wait()  # let the landing screen show first
        try:
            engine.run_benchmark(config_path, cb=state.handle,
                                 model_filter=model_filter)
        except Exception as e:
            error_holder["error"] = str(e)
            with state.lock:
                state.finished = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Use the alternate screen buffer only on a real interactive terminal.
    # When stdout is piped/redirected (tests, CI), fall back to inline mode so
    # the run still completes and prints normally instead of hanging.
    import sys
    use_screen = sys.stdout.isatty()

    tick = 0
    with Live(_render_landing(0), refresh_per_second=12, screen=use_screen) as live:
        # Phase 1: landing.
        landing_end = time.time() + landing_seconds
        while time.time() < landing_end:
            live.update(_render_landing(tick))
            tick += 1
            time.sleep(1 / 12)
        started.set()  # release the worker

        # Phase 2: running (until finished).
        while True:
            live.update(_render_running(state, tick, overall, per_model,
                                        overall_task, model_task))
            tick += 1
            with state.lock:
                done = state.finished
            if done and not t.is_alive():
                break
            time.sleep(1 / 12)

        if "error" in error_holder:
            live.update(Group(Align.center(Panel(
                Text(f"engine error:\n{error_holder['error']}", style="bold red"),
                border_style="red", title="[bold red]:: ABORTED ::"))))
            time.sleep(3)
            return

        # Phase 3: results. Keep animating the GAME OVER / verdict screen until
        # the user presses a key. We key off `use_screen` (the alternate-screen
        # mode) rather than stdin.isatty(): on Windows msvcrt reads the console
        # directly, and an earlier menu prompt can leave sys.stdin looking
        # non-interactive even in a real terminal. When output is piped
        # (no alt screen), fall back to a short timed hold so scripts/tests
        # don't hang.
        if use_screen:
            # Drain any buffered keystrokes so a stray Enter from the menu
            # doesn't instantly dismiss the screen.
            _drain_keys()
            while True:
                live.update(_render_results(state, tick, wait_key=True))
                tick += 1
                if _key_pressed():
                    break
                time.sleep(1 / 12)
        else:
            results_frames = max(int(results_hold_seconds * 12), 12)
            for _ in range(results_frames):
                live.update(_render_results(state, tick, wait_key=False))
                tick += 1
                time.sleep(1 / 12)
                time.sleep(1 / 12)

    # After leaving the alternate screen, print the report to the scrollback
    # and write CSV/Markdown artifacts.
    with state.lock:
        out_dir = state.out_dir
    if out_dir and os.path.exists(os.path.join(out_dir, "results.json")):
        with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as f:
            results = json.load(f)
        report.generate(results, out_dir)
