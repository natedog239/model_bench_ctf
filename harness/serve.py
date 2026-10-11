"""Manual "serve and hold" model-swap controller.

Unlike the benchmark engine (which loads a model, fires every test case at it,
scores, and unloads in a batch loop), this controller does exactly one thing:
load a single model onto a backend, keep it serving on a stable HTTP endpoint,
and let you swap to another model on demand. No scoring, no test suites.

This is the integration point for an EXTERNAL harness (e.g. an XSS benchmark
container) that owns its own orchestration and pass/fail judging: point that
container at the endpoint this controller prints, run your trials, then come
back here and press a key to load the next model behind the same endpoint.

Backends (reuses harness/runner.py as-is):
  ssh    - orchestrate llama-server on a remote GPU box over SSH (RECOMMENDED
           for the 32GB server workflow). Lists the remote library, launches
           one model, health-checks it, holds it; swapping kills it to free
           VRAM before the next.
  server - spawn a local Vulkan llama-server.exe on this machine.
  remote - a llama-server you already started elsewhere; this controller only
           health-checks and reports it (one server hosts one model, so there
           is nothing to swap -- restart it yourself to change models).

Keyboard commands while serving:
  [n] load a different model (unloads the current one first)
  [r] reload the current model
  [u] unload (stop serving, keep the controller open)
  [i] print the current endpoint + model info again
  [q] quit (unloads and exits)
"""

import os
import shlex
import subprocess
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from harness import engine, profiles
from harness.runner import (
    RemoteServerRunner,
    ServerRunner,
    SshServerRunner,
    _ssh_base_cmd,
    ssh_list_models,
)


def _fmt_size(num_bytes: int) -> str:
    mb = (num_bytes or 0) / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"


# ------------------------------------------------------------------ selection

def _resolve_backend(config_path: str, backend_override):
    """Pick the serve backend: an explicit override, else an interactive menu,
    else the config's backend. Returns a concrete backend string."""
    cfg = engine.load_yaml(config_path)
    if backend_override:
        return backend_override
    console = Console()
    if not sys.stdin.isatty():
        return cfg.get("backend", "server")

    ssh_host = (cfg.get("ssh") or {}).get("host", "the remote host")
    remote_url = (cfg.get("remote") or {}).get("base_url", "a running server")
    console.print(
        "\n[bold]Where should the model be served?[/bold]\n"
        "  [bold magenta]L[/bold magenta]) local  - spawn llama-server.exe on this machine\n"
        f"  [bold magenta]R[/bold magenta]) remote - orchestrate llama-server on "
        f"[cyan]{ssh_host}[/cyan] over SSH (swap models on the 32GB box)\n"
        f"  [bold magenta]E[/bold magenta]) existing - a server you already started "
        f"([cyan]{remote_url}[/cyan]); no swapping\n"
        "  ([dim]Enter = remote/SSH[/dim])"
    )
    choice = input("backend > ").strip().lower()
    if choice in ("", "r", "remote", "ssh"):
        return "ssh"
    if choice in ("l", "local", "server"):
        return "server"
    if choice in ("e", "existing"):
        return "remote"
    console.print("[yellow]unrecognized choice; using remote/SSH.[/yellow]")
    return "ssh"


def _list_models(cfg: dict, backend: str):
    """Return a roster of swappable models for the backend.

    Each entry is a dict: {path, size_bytes, name}. For 'remote' there is a
    single fixed model (one server hosts one model), so the roster has one
    synthetic entry.
    """
    if backend == "ssh":
        discovered = ssh_list_models(cfg.get("ssh", {}))
        return [
            {"path": m["path"], "size_bytes": m.get("size_bytes", 0),
             "name": os.path.basename(m["path"])}
            for m in discovered
        ]
    if backend == "remote":
        rc = cfg.get("remote", {}) or {}
        name = rc.get("model_name") or f"remote:{rc.get('base_url', '')}"
        return [{"path": None, "size_bytes": int(rc.get("size_bytes", 0) or 0),
                 "name": name}]
    # local server (and python, though serve only makes sense for server)
    paths = engine.find_models(cfg["models_dir"], cfg.get("exclude_dirs"))
    return [{"path": p, "size_bytes": os.path.getsize(p),
             "name": os.path.basename(p)} for p in paths]


def _pick_model(console: Console, roster, backend: str):
    """Show the roster and prompt for a selection. Returns the chosen entry, or
    None if the user cancels. Non-interactive stdin picks the first model."""
    if not roster:
        console.print("[red]No models available for this backend.[/red]")
        return None

    table = Table(title="[bold magenta]AVAILABLE MODELS",
                  header_style="bold magenta", border_style="magenta")
    table.add_column("#", justify="right")
    table.add_column("Model", overflow="fold")
    table.add_column("Size", justify="right")
    for i, m in enumerate(roster, 1):
        table.add_row(str(i), m["name"], _fmt_size(m["size_bytes"]))
    console.print(table)

    if backend == "remote":
        console.print("[dim]remote backend hosts a single fixed model; nothing to pick.[/dim]")
        return roster[0]

    if not sys.stdin.isatty():
        console.print("[dim]non-interactive stdin: selecting the first model.[/dim]")
        return roster[0]

    while True:
        raw = input(f"model # (1-{len(roster)}, or 'c' to cancel) > ").strip().lower()
        if raw in ("c", "cancel", "q", ""):
            return None
        try:
            idx = int(raw)
        except ValueError:
            console.print("[yellow]enter a number from the list.[/yellow]")
            continue
        if 1 <= idx <= len(roster):
            return roster[idx - 1]
        console.print(f"[yellow]out of range; pick 1-{len(roster)}.[/yellow]")


# ------------------------------------------------------------------ runner build

def _resolve_block(backend: str, cfg: dict, model_entry: dict,
                   selected_profile=None) -> dict:
    """Return the backend block (llama:/ssh:) with the matching profile merged.

    Centralizes profile resolution so both the runner build and the [p]rofile
    display use identical effective settings. `remote` has no per-model block to
    tune (one server, one model) so it is returned unchanged.
    """
    base_key = "ssh" if backend == "ssh" else "llama"
    base = cfg.get(base_key, {}) or {}
    if backend == "remote":
        return dict(base)
    name = model_entry.get("name") or (
        os.path.basename(model_entry["path"]) if model_entry.get("path") else "")
    return profiles.resolve(base, cfg.get("profiles"), name, backend,
                            selected=selected_profile)


def _build_runner(backend: str, cfg: dict, model_entry: dict,
                  selected_profile=None):
    """Construct (but do not load) the appropriate runner for a model entry.

    The backend block is profile-resolved first, so the runner receives already
    effective llama-server settings (base block + any matched/selected profile).
    """
    gen_cfg = cfg.get("generation", {})
    if backend == "ssh":
        eff = _resolve_block(backend, cfg, model_entry, selected_profile)
        return SshServerRunner(
            {"path": model_entry["path"], "size_bytes": model_entry["size_bytes"]},
            eff, gen_cfg,
        )
    if backend == "remote":
        return RemoteServerRunner(cfg.get("remote", {}), gen_cfg)
    if backend == "server":
        eff = _resolve_block(backend, cfg, model_entry, selected_profile)
        return ServerRunner(model_entry["path"], eff, gen_cfg)
    raise ValueError(
        f"Backend '{backend}' cannot serve a held model. Use 'ssh', 'server', "
        f"or 'remote'."
    )


def _endpoint_of(runner, backend: str, cfg: dict) -> str:
    """Best-effort HTTP endpoint the external harness should target."""
    base = getattr(runner, "_base_url", "") or ""
    if backend == "server" and base:
        # ServerRunner binds to 127.0.0.1 only. Warn that a container on another
        # host/namespace will not reach a loopback address.
        return base
    return base


def _print_serving(console: Console, runner, backend: str, cfg: dict,
                   model_entry: dict) -> None:
    endpoint = _endpoint_of(runner, backend, cfg)
    chat_url = f"{endpoint}/v1/chat/completions" if endpoint else "(unknown)"

    body = Text()
    body.append("model     ", style="dim")
    body.append(f"{runner.model_name}\n", style="bold green")
    body.append("size      ", style="dim")
    body.append(f"{_fmt_size(getattr(runner, 'size_bytes', 0))}\n", style="white")
    body.append("backend   ", style="dim")
    body.append(f"{backend}\n", style="cyan")
    load_s = getattr(runner, "load_time_s", 0.0)
    if load_s:
        body.append("load time ", style="dim")
        body.append(f"{load_s:.1f}s\n", style="white")
    body.append("endpoint  ", style="dim")
    body.append(f"{endpoint or '(unknown)'}\n", style="bold yellow")
    body.append("chat API  ", style="dim")
    body.append(f"{chat_url}\n", style="yellow")

    note = None
    if backend == "server":
        note = ("Local server binds to 127.0.0.1 only. A harness in a container "
                "or on another host cannot reach loopback -- use the 'ssh'/'remote' "
                "backend, or front this with a reverse proxy / port forward.")
    elif backend == "remote":
        note = ("Remote hosts a single fixed model; this controller cannot swap "
                "it. Restart the remote server with a different --model to change.")

    console.print(Panel(body, title="[bold green]:: SERVING -- point your harness here ::",
                        border_style="green"))
    if note:
        console.print(f"[dim]{note}[/dim]")


# ------------------------------------------------------------------ live stats

def _server_ctx(base_url: str, api_key=None):
    """Return the context window the server actually loaded, via /props.

    Reads default_generation_settings.n_ctx (the per-slot context). Returns None
    if the endpoint isn't reachable or doesn't report it.
    """
    import requests

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = requests.get(f"{base_url}/props", headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    # n_ctx lives under default_generation_settings on recent builds; fall back
    # to a top-level n_ctx if present.
    dgs = data.get("default_generation_settings", {}) or {}
    return dgs.get("n_ctx") or data.get("n_ctx")


def _probe_stats(base_url: str, api_key=None, n_predict: int = 16):
    """Fire a tiny throwaway completion and read llama.cpp's own `timings`.

    This measures real prefill (prompt eval) and decode throughput on the
    already-running server without needing the server's --metrics flag. Returns
    the timings dict plus token counts, or an error string.
    """
    import requests

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "messages": [
            {"role": "user", "content": "Reply with the single word: ok."},
        ],
        "max_tokens": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,   # measure a real prefill, not a cache hit
    }
    try:
        t0 = time.perf_counter()
        r = requests.post(f"{base_url}/v1/chat/completions",
                          json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        wall = time.perf_counter() - t0
        data = r.json()
    except Exception as e:
        return {"error": str(e)}
    return {
        "timings": data.get("timings", {}) or {},
        "usage": data.get("usage", {}) or {},
        "wall_s": wall,
    }


def _print_stats(console: Console, runner, backend: str, cfg: dict) -> None:
    """Probe the live server for context size + prefill/decode throughput."""
    base_url = getattr(runner, "_base_url", "") or ""
    api_key = getattr(runner, "_api_key", None)
    if not base_url:
        console.print("[yellow]no endpoint to probe (is a model loaded?).[/yellow]")
        return

    console.print("[cyan]probing live server (small throwaway request)...[/cyan]")
    n_ctx = _server_ctx(base_url, api_key)
    res = _probe_stats(base_url, api_key)

    if "error" in res:
        console.print(f"[red]stats probe failed:[/red] {res['error']}")
        return

    t = res["timings"]
    u = res["usage"]
    prefill_tps = t.get("prompt_per_second")
    decode_tps = t.get("predicted_per_second")
    prompt_n = t.get("prompt_n", u.get("prompt_tokens", 0))
    predicted_n = t.get("predicted_n", u.get("completion_tokens", 0))

    body = Text()
    body.append("context window ", style="dim")
    body.append(f"{n_ctx if n_ctx else 'unknown'}\n", style="bold cyan")
    body.append("prefill        ", style="dim")
    body.append(
        f"{prefill_tps:.1f} tok/s" if prefill_tps else "n/a",
        style="bold green")
    body.append(f"  ({prompt_n} prompt tok)\n", style="dim")
    body.append("decode         ", style="dim")
    body.append(
        f"{decode_tps:.1f} tok/s" if decode_tps else "n/a",
        style="bold green")
    body.append(f"  ({predicted_n} gen tok)\n", style="dim")
    body.append("round trip     ", style="dim")
    body.append(f"{res['wall_s']:.2f}s (incl. HTTP)\n", style="white")

    console.print(Panel(body, title="[bold cyan]:: LIVE STATS (sampled) ::",
                        border_style="cyan"))
    console.print("[dim]numbers are from a tiny probe request; your harness's own "
                  "responses carry per-request timings too.[/dim]")


def _tail_remote_log(console: Console, cfg: dict, lines: int = 40) -> None:
    """Tail the remote llama-server log over SSH (ssh backend only).

    The ssh runner redirects the server to /tmp/llama-server-<port>.log, so we
    read the tail of that file. This surfaces per-request prompt-eval/eval
    timings that your external harness is generating against the server.
    """
    ssh_cfg = cfg.get("ssh", {}) or {}
    port = int(ssh_cfg.get("http_port", 8080))
    log = f"/tmp/llama-server-{port}.log"
    try:
        out = subprocess.run(
            _ssh_base_cmd(ssh_cfg) + [f"tail -n {int(lines)} {shlex.quote(log)} 2>/dev/null"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception as e:
        console.print(f"[red]could not tail remote log:[/red] {e}")
        return
    text = out.stdout.strip()
    if not text:
        console.print(f"[yellow]remote log {log} is empty or unreadable "
                      f"(no requests yet?).[/yellow]")
        return
    console.print(Panel(Text(text), title=f"[bold cyan]:: REMOTE LOG (tail {lines}) {log} ::",
                        border_style="cyan"))


# ------------------------------------------------------------------ controller

def _menu_prompt(console: Console, backend: str) -> str:
    opts = r"\[n]ext model  \[r]eload  \[u]nload  \[s]tats  \[p]rofile  \[i]nfo  \[q]uit"
    if backend == "ssh":
        opts = r"\[n]ext  \[r]eload  \[u]nload  \[s]tats  \[p]rofile  \[t]ail-log  \[i]nfo  \[q]uit"
    elif backend == "remote":
        opts = r"\[r]echeck  \[s]tats  \[i]nfo  \[q]uit"
    console.print(f"\n[bold magenta]serve>[/bold magenta] {opts}")
    try:
        return input("command > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"


def _load_and_hold(console: Console, backend: str, cfg: dict, model_entry: dict,
                   selected_profile=None):
    """Build + load a runner, print the endpoint, and return the live runner.
    Returns None on failure (after cleaning up)."""
    runner = _build_runner(backend, cfg, model_entry, selected_profile)
    eff = _resolve_block(backend, cfg, model_entry, selected_profile)
    prof_name = eff.get("_profile_name")
    if prof_name:
        console.print(f"[dim]profile:[/dim] [magenta]{prof_name}[/magenta]")
    elif backend != "remote" and cfg.get("profiles"):
        console.print("[dim]profile:[/dim] [dim](none matched; using base block)[/dim]")
    console.print(f"[cyan]loading[/cyan] {runner.model_name} ...")
    try:
        runner.load()
    except Exception as e:
        console.print(f"[red]failed to load {runner.model_name}:[/red] {e}")
        try:
            runner.close()
        except Exception:
            pass
        return None
    _print_serving(console, runner, backend, cfg, model_entry)
    return runner


def _print_profile(console: Console, backend: str, cfg: dict, model_entry: dict,
                   selected_profile=None) -> None:
    """Show the fully-resolved llama-server flags for the current model.

    This is the 'see any flag I want' view: it renders the exact argv the
    runner builds (base block + matched/selected profile + extra_args), plus
    which profile (if any) is active and what else is available.
    """
    if backend == "remote":
        console.print("[yellow]remote backend hosts a server you started "
                      "elsewhere; its flags aren't managed here.[/yellow]")
        return

    eff = _resolve_block(backend, cfg, model_entry, selected_profile)
    name = eff.get("_profile_name")
    server_bin = eff.get("server_bin", "<server_bin>")
    model_path = model_entry.get("path") or model_entry.get("name") or "<model>"
    http_port = eff.get("http_port", 8080)
    argv = profiles.describe(eff, server_bin, model_path,
                             remote=(backend == "ssh"), http_port=http_port)

    body = Text()
    body.append("model    ", style="dim")
    body.append(f"{model_entry.get('name', '?')}\n", style="bold green")
    body.append("profile  ", style="dim")
    body.append(f"{name or '(none — base block)'}\n", style="bold magenta")
    body.append("command  ", style="dim")
    # One token per line indented, so long flag lists stay readable.
    body.append(f"{os.path.basename(str(server_bin))}", style="white")
    rest = argv[1:]
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok.startswith("-") and i + 1 < len(rest) and not rest[i + 1].startswith("-"):
            body.append(f"\n           {tok} ", style="cyan")
            body.append(rest[i + 1], style="white")
            i += 2
        else:
            body.append(f"\n           {tok}", style="cyan")
            i += 1
    body.append("\n")

    console.print(Panel(body, title="[bold magenta]:: RESOLVED llama-server FLAGS ::",
                        border_style="magenta"))

    profs = cfg.get("profiles") or {}
    if profs:
        avail = ", ".join(
            f"[magenta]{n}[/magenta]" + (f" (match {p.get('match')})" if p.get("match") else "")
            for n, p in profs.items()
        )
        console.print(f"[dim]available profiles:[/dim] {avail}")
        console.print("[dim]type a profile name to switch (reloads), or Enter to keep.[/dim]")
    else:
        console.print("[dim]no profiles defined. Add a top-level 'profiles:' block "
                      "in bench.yaml, or set 'extra_args' on the llama:/ssh: block.[/dim]")


def _unload(console: Console, runner) -> None:
    if runner is None:
        return
    console.print(f"[yellow]unloading[/yellow] {runner.model_name} (freeing VRAM) ...")
    try:
        runner.close()
    except Exception as e:
        console.print(f"[red]error during unload:[/red] {e}")


def run_serve(config_path: str, backend_override=None) -> None:
    """Entry point for `python main.py --serve`.

    Interactive loop that holds one model at a time on a stable endpoint and
    lets the operator swap models without re-running the whole harness.
    """
    console = Console()
    backend = _resolve_backend(config_path, backend_override)
    cfg = engine.load_yaml(config_path)

    console.print(Panel(
        Text.assemble(
            ("Manual model-swap serve mode\n", "bold magenta"),
            (f"backend: {backend}\n", "cyan"),
            ("Load a model, point your external harness at the endpoint, run your "
             "trials, then swap to the next model from here.", "dim"),
        ),
        border_style="magenta", title="[bold magenta]:: SERVE & HOLD ::",
    ))

    try:
        roster = _list_models(cfg, backend)
    except Exception as e:
        console.print(f"[red]could not list models for backend '{backend}':[/red] {e}")
        return

    current = _pick_model(console, roster, backend)
    if current is None:
        console.print("[dim]no model selected; exiting serve mode.[/dim]")
        return

    # Explicit profile override; None = auto-match by model name. Switching
    # profiles live reloads the model with the new flag set.
    selected_profile = None

    runner = _load_and_hold(console, backend, cfg, current, selected_profile)

    try:
        while True:
            cmd = _menu_prompt(console, backend)

            if cmd in ("q", "quit", "exit"):
                break

            if cmd in ("i", "info"):
                if runner is not None:
                    _print_serving(console, runner, backend, cfg, current)
                else:
                    console.print("[dim]nothing is being served right now.[/dim]")
                continue

            if cmd in ("s", "stats"):
                if runner is not None:
                    _print_stats(console, runner, backend, cfg)
                else:
                    console.print("[dim]nothing is being served right now.[/dim]")
                continue

            if cmd in ("p", "profile", "profiles"):
                _print_profile(console, backend, cfg, current, selected_profile)
                if backend != "remote" and (cfg.get("profiles")):
                    choice = input("profile > ").strip()
                    if choice and choice.lower() not in ("c", "cancel", "keep"):
                        if choice not in (cfg.get("profiles") or {}):
                            console.print(f"[yellow]no profile named '{choice}'. "
                                          f"keeping current.[/yellow]")
                        else:
                            selected_profile = choice
                            console.print(f"[cyan]switching to profile[/cyan] "
                                          f"[magenta]{choice}[/magenta] (reloading)...")
                            _unload(console, runner)
                            runner = _load_and_hold(console, backend, cfg,
                                                    current, selected_profile)
                continue

            if cmd in ("t", "tail", "tail-log", "log"):
                if backend != "ssh":
                    console.print("[yellow]log tailing is only available for the "
                                  "'ssh' backend (the remote server's log).[/yellow]")
                else:
                    _tail_remote_log(console, cfg)
                continue

            if cmd in ("u", "unload"):
                _unload(console, runner)
                runner = None
                continue

            if cmd in ("r", "reload", "recheck"):
                _unload(console, runner)
                runner = _load_and_hold(console, backend, cfg, current,
                                        selected_profile)
                continue

            if cmd in ("n", "next"):
                if backend == "remote":
                    console.print("[yellow]remote backend hosts one fixed model; "
                                  "nothing to swap.[/yellow]")
                    continue
                pick = _pick_model(console, roster, backend)
                if pick is None:
                    console.print("[dim]keeping the current model.[/dim]")
                    continue
                _unload(console, runner)
                current = pick
                # New model: drop the explicit profile and auto-match again.
                selected_profile = None
                runner = _load_and_hold(console, backend, cfg, current,
                                        selected_profile)
                continue

            console.print("[yellow]unrecognized command.[/yellow]")
    finally:
        # Always free the model/VRAM on exit.
        _unload(console, runner)
        console.print("[bold magenta]serve mode ended.[/bold magenta]")
