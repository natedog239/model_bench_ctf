# Plan: spin out `llama-serve` as its own repo

## Decision

- **This repo (`Model_Bench_CTF`) keeps the guts.** The CTF benchmark and every
  piece it needs (`engine`, `scoring`, `tests`, `history`, `report`, `tui`, and
  the runners/profiles it currently calls) stay here and keep working unchanged.
- **A brand-new repo (`llama-serve`) is where serve/management development
  continues.** It starts as a fresh project that *carries over* the serve layer,
  not an in-place extraction. The two evolve independently from day one.

This is a **copy-and-diverge**, not a dependency link. The CTF repo will not
import the new repo, and the new repo will not import the CTF repo. We accept a
small amount of duplicated runner/profile code as the price of zero coupling.
(If that duplication ever hurts, we can later promote `llama-serve` to a real
dependency — see "Later: optional convergence".)

## Why this is clean (verified)

The serve layer is already a near-standalone slice:

- `harness/profiles.py` — **zero** project deps (stdlib only: `fnmatch`, `shlex`).
- `harness/runner.py` — stdlib + `requests` + lazy `llama_cpp`; imports
  `harness.profiles` and nothing else from the project.
- `harness/serve.py` — Rich UI + the runners + `profiles`. Its only touch of the
  benchmark is two generic helpers from `engine.py`:
  - `load_yaml(path)` — a 2-line `yaml.safe_load` wrapper.
  - `find_models(models_dir, exclude_dirs)` — generic `.gguf` discovery.

Config keys the serve layer reads (all model-serving, none CTF-specific):
`backend`, `ssh:`, `remote:`, `llama:`, `profiles:`, `generation:`,
`models_dir`, `exclude_dirs`. It never reads `flag`, `tests`, `results_dir`, or
anything from scoring/history/report.

What stays behind because it is benchmark-specific: `engine.py` (run loop,
scoring orchestration), `scoring.py`, `transforms.py`, `history.py`,
`report.py`, `tui.py` (the "Evil Wizard" scoring dashboard), `config/tests/`.

## Target layout of the new repo

```
llama-serve/
  pyproject.toml            # name, deps, console entry point
  README.md                 # lead with the manager story (manager.png)
  LICENSE
  .gitignore
  llama_serve/
    __init__.py
    config.py               # load_yaml + find_models (lifted, no engine import)
    profiles.py             # copied verbatim from harness/profiles.py
    runner.py               # copied; ServerRunner, SshServerRunner,
                            #   RemoteServerRunner, ssh_list_models, helpers
    serve.py                # copied serve-and-hold controller, re-pointed imports
    cli.py                  # argparse front end -> `llama-serve ...`
  config/
    serve.example.yaml      # llama:/ssh:/remote:/profiles:/generation: only
  tests/
    test_profiles.py        # the assertions we already ran, made permanent
```

Package name `llama_serve`, distribution/CLI name `llama-serve` (hyphen).

## Carry-over steps (in the NEW repo)

1. **Scaffold**: `git init llama-serve`, add `pyproject.toml`
   (deps: `PyYAML`, `rich`, `requests`; extra `cpu` = `llama-cpp-python`),
   `.gitignore` (copy this repo's), `LICENSE`.
2. **Copy three files** into `llama_serve/`: `profiles.py`, `runner.py`,
   `serve.py`. Rewrite imports: `from harness import ...` -> `from llama_serve
   import ...`; `from harness.runner import ...` -> `from llama_serve.runner
   import ...`.
3. **Create `llama_serve/config.py`** holding `load_yaml` and `find_models`
   (lifted from `engine.py`, plus the `DEFAULT_EXCLUDE_DIRS` constant). Point
   `serve.py` at `config.load_yaml` / `config.find_models` instead of `engine`.
   This removes the last tie to the benchmark's `engine`.
4. **Add `llama_serve/cli.py`** with a standalone entry point. Minimum parity
   with today's `main.py --serve`:
   - `llama-serve serve [--config C] [--backend ssh|server|remote]`
   - `llama-serve profiles [--config C] [--model NAME]`  (print resolved flags)
   Register it in `pyproject.toml` as `llama-serve = "llama_serve.cli:main"`.
5. **Author `config/serve.example.yaml`** — just the serving blocks
   (`backend`, `llama:`, `ssh:`, `remote:`, `profiles:`, `generation:`,
   `models_dir`, `exclude_dirs`). Drop `flag`, `tests`, `results_dir`.
6. **Move the tests in**: turn the throwaway profile checks we ran (extra_args
   shapes, glob matching, backend restriction, precedence, build_cli output)
   into `tests/test_profiles.py` so the new repo has real coverage from commit 1.
7. **README**: lead with the "Remote llama.cpp Model Manager" framing and
   `manager.png` (copy the asset over). The serve-mode command table and the
   "Flag profiles" section already drafted in this repo's README port directly.
8. **Smoke test**: `pip install -e .[cpu]`, then
   `llama-serve serve --backend server` against a local GGUF; confirm load,
   `[s]tats`, `[p]rofile`, and clean shutdown.

## What (if anything) changes in THIS repo

Nothing is required. The CTF harness keeps its own copies of
`runner.py`/`profiles.py` and continues to work. Optional, cosmetic only:

- Add a short note in this README pointing at the `llama-serve` repo as the
  place that serving/management now lives, so the two projects cross-reference.

We explicitly are **not** deleting the serve code from here, because `serve.py`
and the runners are still used by `main.py --serve` and the benchmark engine.

## Divergence boundary (so the copies don't silently rot)

Keep a one-line provenance comment at the top of each copied file in the new
repo ("Originally forked from Model_Bench_CTF/harness/<file> on <date>."). After
the fork, treat the new repo as the source of truth for serve/profiles. If a bug
fix is made in one repo that matters to the other, port it by hand — the two
files will drift (the new repo grows a CLI/config schema; the CTF copy stays
minimal), so a blind copy-back is not safe.

## Later: optional convergence (only if duplication hurts)

If maintaining two copies becomes a burden, promote `llama-serve` to a published
package and make this repo depend on it:

- Add `llama-serve` to this repo's `requirements.txt` (git URL or PyPI).
- Replace `harness/runner.py` + `harness/profiles.py` with thin re-exports:
  `from llama_serve.runner import *`.
- Keep `harness/engine.py`'s `_make_runner` pointing at the imported classes.

Deferred on purpose: it couples release cadences and adds an install step. Only
do it once there is a second real consumer (e.g. the external XSS harness).

## Open questions for you

1. **Repo name** — `llama-serve`? (used as the pip name + `llama-serve` CLI)
2. **CLI scope for v1** — just `serve` + `profiles`, or also a one-shot
   `llama-serve run --model X --profile Y` non-interactive launcher?
3. **Config file name** — `serve.yaml` (standalone) vs. staying compatible with
   a `bench.yaml` so you can point the new tool at the CTF repo's config too?
4. **License** — reuse this repo's LICENSE as-is?
```
