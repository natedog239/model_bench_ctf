"""CTF small-model benchmarking harness (CLI front-end).

Loads each GGUF model in the models directory, runs the configured test
suites, scores every case, records performance metrics, and writes results
plus a comparison report.

Usage:
  python main.py                     # run everything from config/bench.yaml
  python main.py --config other.yaml # use a different top-level config
  python main.py --dry-run           # validate config + tests without loading models
  python main.py --report-only DIR   # regenerate report from an existing results.json
  python main.py --tui               # launch the live TUI dashboard instead
"""

import argparse
import json
import os
import sys

from harness import engine, report


def _cli_callback(event: str, payload: dict) -> None:
    """Plain-text progress printer for the non-TUI run."""
    if event == "run_start":
        print(f"Run {payload['run_id']}: {payload['model_count']} model(s), "
              f"{payload['case_count']} case(s) each -> {payload['out_dir']}")
    elif event == "model_start":
        print(f"\n=== [{payload['index']}/{payload['total']}] "
              f"{payload['model_name']} ({payload['size_bytes'] / 1e6:.0f} MB) ===")
    elif event == "model_loaded":
        print(f"    loaded in {payload['load_time_s']:.2f}s")
    elif event == "model_error":
        print(f"    ERROR: {payload['error']}", file=sys.stderr)
    elif event == "test_start":
        print(f"  -> {payload['test_id']}")
    elif event == "case_done":
        status = "PASS" if payload["passed"] else "FAIL"
        tps = payload["stats"].get("tokens_per_s", 0)
        print(f"       [{status}] {payload['case_id']}  ({tps} tok/s)")
    elif event == "model_done":
        print(f"    {payload['model_name']}: {payload['passed']}/{payload['total']} passed")


def main():
    parser = argparse.ArgumentParser(description="CTF small-model benchmark harness")
    parser.add_argument("--config", default="config/bench.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and tests without loading models.")
    parser.add_argument("--report-only", metavar="DIR",
                        help="Regenerate report from an existing results.json directory.")
    parser.add_argument("--tui", action="store_true",
                        help="Launch the live TUI dashboard.")
    parser.add_argument("--serve", action="store_true",
                        help="Manual model-swap serve mode: load one model, hold it on a "
                             "stable endpoint for an external harness, swap on demand.")
    parser.add_argument("--backend", choices=["server", "python", "remote", "ssh"],
                        help="Override the config's backend for this run "
                             "(e.g. 'ssh' to orchestrate a remote GPU box).")
    parser.add_argument("--ingest", metavar="DIR",
                        help="Index an existing run's results.json into the history DB, then exit.")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Print models ranked by score from the history DB (with on-disk flag), then exit.")
    parser.add_argument("--filter-lt", type=int, metavar="X",
                        help="Only run models whose last total score was < X.")
    parser.add_argument("--filter-gt", type=int, metavar="X",
                        help="Only run models whose last total score was > X.")
    parser.add_argument("--filter-untested", action="store_true",
                        help="Only run models that have never been benchmarked.")
    args = parser.parse_args()

    if args.report_only:
        with open(os.path.join(args.report_only, "results.json"), encoding="utf-8") as f:
            results = json.load(f)
        report.generate(results, args.report_only)
        return

    if args.ingest:
        from harness import history
        with open(os.path.join(args.ingest, "results.json"), encoding="utf-8") as f:
            results = json.load(f)
        config_dir = os.path.dirname(os.path.abspath(args.config))
        history.ingest_results(results, config_dir=config_dir)
        print(f"Ingested run {results['run_id']} into the history DB.")
        return

    if args.leaderboard:
        from harness import history
        _cfg = engine.load_yaml(args.config)
        history.print_leaderboard(_cfg["models_dir"], exclude_dirs=_cfg.get("exclude_dirs"))
        return

    if args.serve:
        from harness import serve
        serve.run_serve(args.config, backend_override=args.backend)
        return

    # Resolve a CLI score filter into a model list (shared by TUI + CLI paths).
    def _cli_model_filter():
        if (args.filter_lt is None and args.filter_gt is None
                and not args.filter_untested):
            return None
        from harness import history
        _c = engine.load_yaml(args.config)
        _be = args.backend or _c.get("backend", "server")
        if _be == "ssh":
            from harness.runner import ssh_list_models
            all_paths = [m["path"] for m in ssh_list_models(_c.get("ssh", {}))]
        else:
            all_paths = engine.find_models(_c["models_dir"], _c.get("exclude_dirs"))
        if args.filter_untested:
            sel, _ = history.filter_models(all_paths, "untested")
        elif args.filter_lt is not None:
            sel, _ = history.filter_models(all_paths, "lt", args.filter_lt)
        else:
            sel, _ = history.filter_models(all_paths, "gt", args.filter_gt)
        return sel

    _has_cli_filter = (args.filter_lt is not None or args.filter_gt is not None
                       or args.filter_untested)

    if args.tui:
        from harness.tui import run_tui, select_models_menu, select_backend_menu
        # 1) Choose where to run (local vs remote/SSH). --backend overrides the
        #    menu; otherwise the menu offers a choice (None = use config).
        backend = args.backend or select_backend_menu(args.config)
        # The model roster/discovery must match the backend actually used.
        cfg = engine.load_yaml(args.config)
        effective_backend = backend or cfg.get("backend", "server")
        # 2) Choose which models. CLI filter flags take precedence; otherwise
        #    show the interactive (backend-aware) model menu.
        mf = _cli_model_filter()
        if mf is None and not _has_cli_filter:
            mf = select_models_menu(args.config, backend=effective_backend)
        run_tui(args.config, model_filter=mf, backend_override=backend)
        return

    cfg = engine.load_yaml(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    tests = engine.load_tests(config_dir, cfg["tests"], cfg["flag"])
    effective_backend = args.backend or cfg.get("backend", "server")

    print(f"Loaded {len(tests)} test suite(s): {[t['id'] for t in tests]}")
    total_cases = sum(len(t["cases"]) for t in tests)
    print(f"Total cases per model: {total_cases}")

    if effective_backend == "ssh":
        from harness.runner import ssh_list_models
        discovered = ssh_list_models(cfg.get("ssh", {}))
        models = [m["path"] for m in discovered]
        src = f"remote host '{cfg.get('ssh', {}).get('host')}'"
    elif effective_backend == "remote":
        models = ["<single remote server>"]
        src = "the configured remote server"
    else:
        models = engine.find_models(cfg["models_dir"], cfg.get("exclude_dirs"))
        src = f"'{cfg['models_dir']}'"
    print(f"Found {len(models)} model(s) via {src} (backend: {effective_backend})")

    if args.dry_run:
        print("\n[dry-run] Config and tests validated. No models loaded.")
        for t in tests:
            print(f"  - {t['id']}: {len(t['cases'])} cases")
        return

    if not models:
        print(f"\nNo .gguf models found via {src}. "
              f"Add model files and rerun.", file=sys.stderr)
        sys.exit(1)

    results = engine.run_benchmark(args.config, cb=_cli_callback,
                                   model_filter=_cli_model_filter(),
                                   backend_override=args.backend)
    report.generate(results, os.path.join(cfg["results_dir"], results["run_id"]))


if __name__ == "__main__":
    main()
