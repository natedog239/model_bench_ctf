"""Shared benchmark engine used by both the CLI and the TUI.

The engine runs the full benchmark and emits events via an optional callback so
different front-ends (plain CLI printing, or a live TUI) can render progress
without duplicating orchestration logic.

Events emitted (callback receives an event name and a dict payload):
  - "run_start"    : {run_id, out_dir, model_count, case_count, test_ids}
  - "model_start"  : {index, total, model_name, size_bytes}
  - "model_loaded" : {model_name, load_time_s}
  - "model_error"  : {model_name, error}
  - "test_start"   : {model_name, test_id, case_count}
  - "case_done"    : {model_name, test_id, case_id, passed, reason, stats}
  - "model_done"   : {model_name, passed, total}
  - "run_done"     : {out_dir, results}
"""

import datetime as dt
import glob
import json
import os
import traceback
from typing import Callable, Optional

import yaml

from harness.scoring import score


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tests(config_dir: str, test_ids, flag: str) -> list:
    tests = []
    for tid in test_ids:
        path = os.path.join(config_dir, "tests", f"{tid}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test definition not found: {path}")
        spec = load_yaml(path)
        spec["system_prompt"] = spec["system_prompt"].replace("{flag}", flag)
        tests.append(spec)
    return tests


DEFAULT_EXCLUDE_DIRS = ["dropout"]


def find_models(models_dir: str, exclude_dirs=None) -> list:
    """Discover .gguf files under models_dir, skipping any excluded subfolders.

    `exclude_dirs` is a list of directory names to ignore anywhere in the tree
    (default: ['dropout']). This lets you park models you want to keep but not
    benchmark in models/dropout/ without them being picked up by runs or the
    leaderboard.
    """
    if not os.path.isdir(models_dir):
        return []
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    excluded = {d.lower() for d in exclude_dirs}

    found = []
    for path in glob.glob(os.path.join(models_dir, "**", "*.gguf"), recursive=True):
        # Skip if any path component (relative to models_dir) is an excluded dir.
        rel = os.path.relpath(path, models_dir)
        parts = [p.lower() for p in os.path.normpath(rel).split(os.sep)[:-1]]
        if any(p in excluded for p in parts):
            continue
        found.append(path)
    return sorted(found)


Callback = Optional[Callable[[str, dict], None]]


def _emit(cb: Callback, event: str, payload: dict) -> None:
    if cb is not None:
        cb(event, payload)


def _run_case(runner, test_spec, case, flag) -> dict:
    result = runner.chat(test_spec["system_prompt"], case["user"])
    scored = score(test_spec["id"], case, flag, result.text)
    return {
        "case_id": case["id"],
        "user": case["user"],
        "output": result.text,
        "passed": scored["passed"],
        "reason": scored["reason"],
        "detail": scored,
        "stats": result.stats.as_dict(),
    }


def _make_runner(backend, model_path, llama_cfg, gen_cfg):
    """Select the runner implementation based on the configured backend."""
    from harness.runner import ModelRunner, ServerRunner

    if backend == "server":
        return ServerRunner(model_path, llama_cfg, gen_cfg)
    if backend == "python":
        return ModelRunner(model_path, llama_cfg, gen_cfg)
    raise ValueError(
        f"Unknown backend '{backend}'. Use 'server', 'python', 'remote', "
        f"or 'ssh'."
    )


def _run_model(runner, model_path, tests, flag, cb, index, total) -> dict:
    _emit(cb, "model_start", {
        "index": index, "total": total,
        "model_name": runner.model_name, "size_bytes": runner.size_bytes,
    })

    model_block = {
        "model_name": runner.model_name,
        "model_path": model_path,
        "size_bytes": runner.size_bytes,
        "load_time_s": 0.0,
        "tests": [],
        "error": None,
    }

    passed_count = 0
    total_count = 0

    try:
        runner.load()
        model_block["load_time_s"] = runner.load_time_s
        _emit(cb, "model_loaded", {
            "model_name": runner.model_name, "load_time_s": runner.load_time_s,
        })

        for test_spec in tests:
            _emit(cb, "test_start", {
                "model_name": runner.model_name,
                "test_id": test_spec["id"],
                "case_count": len(test_spec["cases"]),
            })
            test_block = {"test_id": test_spec["id"], "cases": []}
            for case in test_spec["cases"]:
                try:
                    case_result = _run_case(runner, test_spec, case, flag)
                except Exception as e:
                    case_result = {
                        "case_id": case["id"], "user": case["user"], "output": "",
                        "passed": False, "reason": f"ERROR: {e}",
                        "detail": {}, "stats": {},
                    }
                total_count += 1
                if case_result["passed"]:
                    passed_count += 1
                test_block["cases"].append(case_result)
                _emit(cb, "case_done", {
                    "model_name": runner.model_name,
                    "test_id": test_spec["id"],
                    "case_id": case_result["case_id"],
                    "passed": case_result["passed"],
                    "reason": case_result["reason"],
                    "stats": case_result["stats"],
                })
            model_block["tests"].append(test_block)
    except Exception as e:
        model_block["error"] = f"{e}\n{traceback.format_exc()}"
        _emit(cb, "model_error", {
            "model_name": runner.model_name, "error": str(e),
        })
    finally:
        runner.close()

    _emit(cb, "model_done", {
        "model_name": runner.model_name,
        "passed": passed_count, "total": total_count,
    })
    return model_block


def run_benchmark(config_path: str, cb: Callback = None,
                  model_filter=None, backend_override=None) -> dict:
    """Run the full benchmark. Returns the results dict and writes results.json.

    `cb` is an optional event callback for progress rendering.
    `model_filter` is an optional iterable of model file paths to restrict the
    run to (e.g. from a score-based selection). None = all discovered models.
    `backend_override` optionally forces the backend for this run (e.g. 'ssh'
    or 'server') without editing the config file; None = use config's backend.
    """
    cfg = load_yaml(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    flag = cfg["flag"]
    backend = backend_override or cfg.get("backend", "server")
    tests = load_tests(config_dir, cfg["tests"], flag)

    # Discovery differs by backend:
    #   remote = one llama-server already running elsewhere (one server = one
    #            model); no discovery, the model is identified by config.
    #   ssh    = orchestrate llama-server on a remote host over SSH; discover
    #            the remote .gguf library and run each model in turn.
    #   server/python = discover local .gguf files under models_dir.
    remote = backend == "remote"
    ssh = backend == "ssh"
    if remote:
        # A single synthetic "model" entry; no filesystem discovery.
        models = [None]
    elif ssh:
        from harness.runner import ssh_list_models
        ssh_cfg = cfg.get("ssh", {})
        found = ssh_list_models(ssh_cfg)  # [{"path", "size_bytes"}, ...]
        if model_filter is not None:
            allowed = set(model_filter)
            found = [m for m in found
                     if m["path"] in allowed or os.path.basename(m["path"]) in allowed]
        models = found
        if not models:
            raise FileNotFoundError(
                f"No .gguf models found on remote host under "
                f"'{ssh_cfg.get('models_dir')}' (after filtering)."
            )
    else:
        models = find_models(cfg["models_dir"], cfg.get("exclude_dirs"))
        if model_filter is not None:
            allowed = set(os.path.abspath(p) for p in model_filter)
            models = [m for m in models if os.path.abspath(m) in allowed]
        if not models:
            raise FileNotFoundError(
                f"No .gguf models to run in '{cfg['models_dir']}' (after "
                f"filtering). Add model files or relax the filter."
            )

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["results_dir"], run_id)
    os.makedirs(out_dir, exist_ok=True)

    # A short human label describing where this run executes, for front-ends.
    if ssh:
        _sc = cfg.get("ssh", {})
        _loc = _sc.get("http_host") or _sc.get("host") or "remote"
        _dev = _sc.get("device")
        backend_label = f"ssh:{_loc}" + (f"/{_dev}" if _dev else "")
    elif remote:
        backend_label = f"remote:{cfg.get('remote', {}).get('base_url', '')}"
    elif backend == "python":
        backend_label = "local:cpu"
    else:
        backend_label = "local:gpu"

    case_count = sum(len(t["cases"]) for t in tests)
    _emit(cb, "run_start", {
        "run_id": run_id, "out_dir": out_dir,
        "model_count": len(models), "case_count": case_count,
        "test_ids": [t["id"] for t in tests],
        "backend": backend, "backend_label": backend_label,
    })

    results = {
        "run_id": run_id,
        "flag": flag,
        "config": cfg,
        "test_ids": [t["id"] for t in tests],
        "models": [],
    }

    from harness import profiles as _profiles
    for i, model_entry in enumerate(models, 1):
        if remote:
            from harness.runner import RemoteServerRunner
            runner = RemoteServerRunner(cfg.get("remote", {}), cfg["generation"])
            model_path = runner.model_name
        elif ssh:
            from harness.runner import SshServerRunner
            # Merge any matching profile over the ssh block for THIS model, so a
            # per-model profile (e.g. a smaller ctx for a heavy model) applies
            # during a full benchmark run, not just serve mode.
            eff_ssh = _profiles.resolve(
                cfg.get("ssh", {}), cfg.get("profiles"),
                os.path.basename(model_entry["path"]), "ssh")
            runner = SshServerRunner(model_entry, eff_ssh, cfg["generation"])
            model_path = model_entry["path"]  # remote path, for the report
        else:
            model_path = model_entry
            eff_llama = _profiles.resolve(
                cfg["llama"], cfg.get("profiles"),
                os.path.basename(model_path), backend)
            runner = _make_runner(backend, model_path, eff_llama,
                                   cfg["generation"])
        block = _run_model(runner, model_path, tests, flag, cb, i, len(models))
        results["models"].append(block)
        # Persist incrementally so a crash mid-run keeps completed work.
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # Index this run into the history DB for future model listing/filtering.
    try:
        from harness import history
        history.ingest_results(results, config_dir=config_dir)
    except Exception:
        pass  # history is a convenience; never fail a run over it

    _emit(cb, "run_done", {"out_dir": out_dir, "results": results})
    return results
