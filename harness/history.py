"""SQLite run-history layer.

Stores a queryable index of past runs so we can list models with their prior
performance and filter which models to run next (all / score < X / score > X).

Deliberately does NOT store per-case pass/fail rows -- those live in each run's
results.json archive. This DB keeps just enough to answer "how did model X do
last time" and to compare runs/configs over time.

Tables:
  runs         : one row per benchmark run (config snapshot + prompt versions)
  model_scores : one row per (run, model) with the aggregate score + perf
"""

import hashlib
import json
import os
import sqlite3
from typing import Optional

DEFAULT_DB = "bench.db"


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id          TEXT PRIMARY KEY,
                created_ts      TEXT,
                backend         TEXT,
                flag            TEXT,
                llama_cfg       TEXT,   -- JSON
                gen_cfg         TEXT,   -- JSON
                test_ids        TEXT,   -- JSON list
                prompt_versions TEXT    -- JSON {test_id: sha256[:12]}
            );

            CREATE TABLE IF NOT EXISTS model_scores (
                run_id       TEXT,
                model_name   TEXT,
                size_bytes   INTEGER,
                status       TEXT,       -- ok | error
                total_pass   INTEGER,
                total_cases  INTEGER,
                avg_tps      REAL,
                load_time_s  REAL,
                PRIMARY KEY (run_id, model_name),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            """
        )


def _prompt_versions(cfg: dict, config_dir: str) -> dict:
    """Compute a stable short hash of each test's resolved system prompt.

    Reads the test YAML and hashes the raw system_prompt template (pre-flag
    substitution), so a prompt edit changes the version even if the flag is the
    same.
    """
    versions = {}
    try:
        import yaml
    except Exception:
        return versions
    for tid in cfg.get("tests", []):
        path = os.path.join(config_dir, "tests", f"{tid}.yaml")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                spec = yaml.safe_load(f)
            prompt = spec.get("system_prompt", "")
            h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            versions[tid] = h
        except Exception:
            continue
    return versions


def ingest_results(results: dict, db_path: str = DEFAULT_DB,
                   config_dir: Optional[str] = None) -> None:
    """Insert (or replace) one run's aggregate data into the DB."""
    init_db(db_path)
    cfg = results.get("config", {}) or {}
    run_id = results["run_id"]

    prompt_versions = _prompt_versions(cfg, config_dir) if config_dir else {}

    with _connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, created_ts, backend, flag, llama_cfg, gen_cfg,
                test_ids, prompt_versions)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                run_id,
                _run_id_to_ts(run_id),
                cfg.get("backend", ""),
                results.get("flag", ""),
                json.dumps(cfg.get("llama", {})),
                json.dumps(cfg.get("generation", {})),
                json.dumps(results.get("test_ids", [])),
                json.dumps(prompt_versions),
            ),
        )

        for m in results.get("models", []):
            total_pass = 0
            total_cases = 0
            tps_vals = []
            for t in m.get("tests", []):
                for c in t.get("cases", []):
                    total_cases += 1
                    if c.get("passed"):
                        total_pass += 1
                    tps = (c.get("stats", {}) or {}).get("tokens_per_s", 0)
                    if tps:
                        tps_vals.append(tps)
            avg_tps = round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else 0.0
            status = "error" if m.get("error") else "ok"

            conn.execute(
                """INSERT OR REPLACE INTO model_scores
                   (run_id, model_name, size_bytes, status, total_pass,
                    total_cases, avg_tps, load_time_s)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    run_id, m["model_name"], m.get("size_bytes", 0), status,
                    total_pass, total_cases, avg_tps, m.get("load_time_s", 0.0),
                ),
            )


def _run_id_to_ts(run_id: str) -> str:
    """Turn a run_id like 20260906_235229 into an ISO-ish timestamp string."""
    try:
        d, t = run_id.split("_")
        return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
    except Exception:
        return run_id


def latest_scores(db_path: str = DEFAULT_DB) -> dict:
    """Return {model_name: {score, total, run_id, ts, avg_tps, status}} using
    each model's most recent run."""
    if not os.path.exists(db_path):
        return {}
    out = {}
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT ms.*, r.created_ts
               FROM model_scores ms JOIN runs r ON ms.run_id = r.run_id
               ORDER BY r.created_ts ASC"""
        ).fetchall()
    for row in rows:
        # later rows overwrite earlier -> ends up as the latest per model
        out[row["model_name"]] = {
            "score": row["total_pass"],
            "total": row["total_cases"],
            "run_id": row["run_id"],
            "ts": row["created_ts"],
            "avg_tps": row["avg_tps"],
            "status": row["status"],
        }
    return out


def filter_models(all_model_paths, mode: str, threshold: int = 0,
                  db_path: str = DEFAULT_DB):
    """Given discovered model paths, return the subset to run.

    mode:
      "all"      -> every model (threshold ignored)
      "lt"       -> models whose LAST total score < threshold (never-run count
                    as eligible, since they have no score yet)
      "gt"       -> models whose LAST total score > threshold (never-run excluded)
      "untested" -> only models with NO record in the history DB (never
                    benchmarked). Models that ran and errored are excluded --
                    they were tested, they just failed.

    Returns (selected_paths, info) where info maps model_name -> reason string
    for display.
    """
    scores = latest_scores(db_path)
    selected = []
    info = {}
    for path in all_model_paths:
        name = os.path.basename(path)
        rec = scores.get(name)
        last = rec["score"] if rec else None

        if mode == "all":
            selected.append(path)
            info[name] = "all"
        elif mode == "lt":
            if last is None or last < threshold:
                selected.append(path)
                info[name] = "never run" if last is None else f"{last} < {threshold}"
        elif mode == "gt":
            if last is not None and last > threshold:
                selected.append(path)
                info[name] = f"{last} > {threshold}"
        elif mode == "untested":
            if rec is None:
                selected.append(path)
                info[name] = "untested"
        else:
            raise ValueError(f"Unknown filter mode '{mode}'")
    return selected, info


def ranked_models(models_dir: str, db_path: str = DEFAULT_DB, exclude_dirs=None):
    """Return models ranked by their latest total score (desc), each annotated
    with whether the .gguf still exists in `models_dir`.

    Rows: {model_name, score, total, avg_tps, status, ts, run_id, present}
    Includes models that are on disk but have never run (score None), and models
    in the DB whose files are now gone (present=False).
    """
    scores = latest_scores(db_path)

    # Model files currently on disk (basename -> full path). Uses the same
    # discovery as runs, so anything parked in an excluded folder (e.g.
    # models/dropout/) is treated as NOT present -> shows as dropped/gone.
    from harness.engine import find_models
    on_disk = {os.path.basename(p): p for p in find_models(models_dir, exclude_dirs)}

    names = set(scores) | set(on_disk)
    rows = []
    for name in names:
        rec = scores.get(name)
        rows.append({
            "model_name": name,
            "score": rec["score"] if rec else None,
            "total": rec["total"] if rec else None,
            "avg_tps": rec["avg_tps"] if rec else None,
            "status": rec["status"] if rec else "never run",
            "ts": rec["ts"] if rec else "-",
            "run_id": rec["run_id"] if rec else "-",
            "present": name in on_disk,
            "path": on_disk.get(name),
        })

    # Rank: highest score first; never-run (None) sink to the bottom; ties by name.
    rows.sort(key=lambda r: (-(r["score"] if r["score"] is not None else -1),
                             r["model_name"].lower()))
    return rows


def print_leaderboard(models_dir: str, db_path: str = DEFAULT_DB,
                      exclude_dirs=None) -> None:
    """Print a DB-driven leaderboard ranked by score, flagging whether each
    model is still present in the models directory (models in excluded folders
    such as models/dropout/ are treated as not present)."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()

    if not os.path.exists(db_path):
        console.print(f"[yellow]No history database at '{db_path}'. "
                      f"Run a benchmark first (or --ingest a past run).[/yellow]")
        return

    rows = ranked_models(models_dir, db_path, exclude_dirs)
    if not rows:
        console.print("[yellow]No models in history or on disk.[/yellow]")
        return

    def _fmt_size(path):
        if not path or not os.path.exists(path):
            return "-"
        mb = os.path.getsize(path) / (1024 * 1024)
        return f"{mb/1024:.2f} GB" if mb >= 1024 else f"{mb:.0f} MB"

    table = Table(title="[bold magenta]MODEL LEADERBOARD (from history DB)",
                  header_style="bold magenta", border_style="magenta")
    table.add_column("#", justify="right")
    table.add_column("Model", overflow="fold")
    table.add_column("Score", justify="right")
    table.add_column("Avg tok/s", justify="right")
    table.add_column("Last Run")
    table.add_column("On Disk", justify="center")

    present_count = 0
    missing_count = 0
    for i, r in enumerate(rows, 1):
        if r["score"] is None:
            score_txt = Text("never run", style="dim")
        elif r["status"] == "error":
            score_txt = Text("ERROR", style="red")
        else:
            score_txt = Text(f"{r['score']}/{r['total']}", style="green")

        if r["present"]:
            present = Text("YES", style="green")
            present_count += 1
        else:
            present = Text("gone", style="bold red")
            missing_count += 1

        table.add_row(
            str(i), r["model_name"], score_txt,
            (str(r["avg_tps"]) if r["avg_tps"] else "-"),
            r["ts"], present,
        )

    console.print(table)
    console.print(
        f"[dim]{len(rows)} model(s) total | "
        f"[green]{present_count} on disk[/green] | "
        f"[red]{missing_count} gone (in history only)[/red][/dim]"
    )
    if missing_count:
        console.print("[dim]'gone' = scored in a past run but no longer in the "
                      "models directory.[/dim]")
