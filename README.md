# CTF Small-Model Benchmark Harness

A lightweight, fully automated harness for evaluating small GGUF models as the
engine behind an AI-focused CTF. It runs each model through
[llama.cpp](https://github.com/ggml-org/llama.cpp), executes a series of tests,
records performance (including tokens/s), and produces a comparison report to
find the smallest model that passes everything.

The recommended runtime is a **prebuilt Vulkan release of llama.cpp** — this
gives GPU acceleration on AMD cards (including the RX 9070 / RDNA4) on Windows
with **no compiler and no SDK to install**. A CPU-only path via the
`llama-cpp-python` bindings is also supported for machines without a GPU.

## Session model

Matching the CTF, **every test case is a fresh, independent chat session**. Each
case sends only a `[system, user]` message pair, and the model's KV cache is
reset before every generation, so no context or history ever carries between
cases. There is no multi-turn state.

## What it tests

**Test 1 — Authentication gating.** The system prompt holds a secret flag and
instructs the model to only reveal it when the user explicitly states they are
authenticated. The harness sends a battery of authenticated and unauthenticated
turns (including social-engineering and prompt-injection attempts). Scoring is
pure string matching: the flag must appear only in the authenticated cases.

**Test 2 — String manipulation (redaction bypass).** Redaction is enforced
**externally** by the chat interface — it intercepts the agent's reply and blocks
the exact flag string, so the user must try again. The model is *not* asked to
self-censor, so its system prompt is the same auth-gatekeeper prompt as Test 1.
Each user turn states authentication and asks for a string transformation
(hyphens between characters, reversed, base64, etc.). A case **passes** when the
model produces the correctly transformed flag (which slips past the exact-match
redaction). Outputting only the raw, untransformed flag is a fail — the external
redaction would block it.

**Test 3 — Tool use (MCP).** Skipped for now. Placeholder left for later.

## Layout

```
config/
  bench.yaml                      # top-level config: backend, models dir, flag, llama + gen params
  tests/
    test1_auth_gating.yaml         # Test 1 cases (data-driven, add freely)
    test2_string_manipulation.yaml # Test 2 cases + transform names
harness/
  engine.py                       # shared orchestration; emits progress events; picks backend
  runner.py                       # ServerRunner (Vulkan llama-server) + ModelRunner (CPU) + metrics
  history.py                      # SQLite run-history index + model score filtering
  scoring.py                      # per-test scorers
  transforms.py                   # flag transformation functions (Test 2)
  report.py                       # CSV + Markdown report generation
  tui.py                          # styled 3-phase live dashboard (rich)
  art.py                          # ASCII art + animation assets for the TUI
tests/
  test_scoring.py                 # unit tests for scoring/transforms (no model needed)
models/                           # <- drop your .gguf files here (immutable library)
results/                          # timestamped run outputs
bench.db                          # SQLite run-history index (auto-created)
setup.ps1                         # venv + Python dependency installer
main.py                           # CLI + TUI entry point
```

## Tool stack

- **Inference:** [llama.cpp](https://github.com/ggml-org/llama.cpp) — a prebuilt
  **Windows Vulkan release** is the recommended runtime (GPU acceleration on
  AMD, no compiler/SDK needed). The harness launches `llama-server.exe` per
  model and talks to its OpenAI-compatible HTTP API.
- **GPU:** AMD Radeon RX 9070 (RDNA4) via the Vulkan backend. The AMD driver
  already ships the Vulkan runtime, so no extra install is required.
- **CPU fallback:** `llama-cpp-python` (prebuilt CPU wheel) via the `python`
  backend, for machines without a GPU.
- **Harness:** Python 3.x in a virtual environment (`rich` TUI, `requests`,
  `PyYAML`, `tabulate`).

## Setup

### 1. Python environment

From the project root, in PowerShell:

```
./setup.ps1          # venv + Python deps + prebuilt CPU llama-cpp-python (fallback)
./setup.ps1 -NoCpu   # venv + Python deps only (skip the CPU backend)
```

This creates `.venv` and installs the Python dependencies. No compiler or SDK
is required — the CPU backend uses a prebuilt wheel. Activate the environment in
later sessions with:

```
.\.venv\Scripts\Activate.ps1
```

To install the Python side manually instead:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. GPU inference (AMD RX 9070 on Windows) — recommended

The RX 9070 is an RDNA4 card. ROCm/HIP support on Windows is limited, so the
reliable path is a **prebuilt Vulkan release of llama.cpp**. This needs **no
compiler and no Vulkan SDK** — your AMD driver already provides the Vulkan
runtime.

1. Download the Windows Vulkan release from the
   [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) —
   the asset named `llama-<build>-bin-win-vulkan-x64.zip`.
2. Extract it into the project folder (e.g.
   `llama-b10830-bin-win-vulkan-x64/`). It contains `llama-server.exe` plus
   `ggml-vulkan.dll` (the GPU backend) and its DLLs.
3. In `config/bench.yaml`, set:
   - `backend: server`
   - `llama.server_bin:` the path to the extracted `llama-server.exe`
   - `llama.n_gpu_layers: 999` (offload all layers; `999` or `-1` = all)

The harness launches one `llama-server.exe` per model, offloads to the GPU,
runs every case against its HTTP API, records llama.cpp's own tokens/s timing,
then shuts the server down before loading the next model.

To confirm the GPU is seen, run `vulkaninfo --summary` — it should list your
`AMD Radeon RX 9070`. On the smallest test model this backend runs ~250+ tok/s
versus ~30 tok/s on CPU.

### 3. CPU fallback (no GPU)

Set `backend: python` in `config/bench.yaml` to use the in-process
`llama-cpp-python` bindings instead. `n_gpu_layers: 0` keeps it on CPU.

## Usage

```
# Validate config and test definitions without loading any model
python main.py --dry-run

# Run the full benchmark over every .gguf in the models directory
python main.py

# Regenerate the report from a previous run
python main.py --report-only results/20260906_120000

# Launch the live TUI dashboard instead of plain CLI output
python main.py --tui
```

### TUI dashboard

`--tui` runs the exact same benchmark (same engine, same results) but wraps it
in a styled, magenta "Evil Wizard" dashboard with three phases:

1. **Landing** — the *Archmage of Benchmarks* with a skull face and arms raised,
   channeling power over a shifting field of magical runes, with a pulsing title
   and a "charging power" bar, on a `:: SUMMONING CIRCLE ::` panel.
2. **Running** — a battle scene: the wizard on the **left** hurls spells and
   lightning at the current model (named, with live status/score) on the
   **right**, beside live progress bars, the `GRIMOIRE OF TARGETS` leaderboard,
   and a `SPELLS CAST` log of recent case results.
3. **Results** — a big `::: GAME OVER :::` screen with a pulsing GAME OVER
   banner, the Archmage's verdict (`A CHAMPION SURVIVES` if a model passed
   everything, otherwise `THE ARCHMAGE REMAINS UNDEFEATED`), the full results
   table printed right in the TUI, a casualties tally, and a breakdown of the
   fallen.

**Failure surfacing.** The dashboard makes failures obvious at every stage:

- The running header shows a live tally — `[OK] all clear so far` until the
  first failure, then a blinking `[!] FAILURES: N case(s) across M model(s)`
  (including any model load errors).
- The leaderboard has a status marker column (`+` all-pass, `!` has failures,
  `X` load error) and a dedicated `Fails` count column, with failing scores in
  red.
- The results screen prints a `[!] FAILURES DETECTED` / `[OK] NO FAILURES`
  tally and, when there are failures, a `:: BREACH FAILURES ::` panel listing
  each failing model and its failed cases (auth leaks are tagged `[LEAK]`).

On completion it writes the same `results.json`, `cases.csv`, and `report.md` as
the CLI path. The art lives in `harness/art.py` so you can retheme it freely.

The dashboard uses the terminal's alternate screen when attached to a real TTY,
and automatically falls back to inline rendering when output is piped.

## Outputs (per run, under `results/<timestamp>/`)

- `results.json` — full structured results (every case: prompt, output, pass/fail,
  reason, and per-call token/latency/throughput stats). Written incrementally.
- `cases.csv` — one row per (model, test, case) for spreadsheets.
- `report.md` — leaderboard ranking models by all-pass, then size, then speed,
  with the best (smallest passing) model called out.

## Run history and model selection

Every run is also indexed into a small SQLite database (`bench.db`) via
`harness/history.py`. It stores each run's config snapshot (backend, llama and
generation params, flag, and a hash of each test's system prompt) plus each
model's aggregate score, speed, and status. Per-case pass/fail detail stays in
the JSON archive — the DB is just the queryable cross-run index.

`models/` is never modified — it stays the immutable library. Selection happens
at run time from the DB, so nothing is copied or moved.

**Benching models (dropout folder).** To keep a `.gguf` but exclude it from runs
and the leaderboard, move it into a subfolder listed under `exclude_dirs` in
`config/bench.yaml` (default: `dropout`), e.g. `models/dropout/`. Excluded
models are skipped by model discovery everywhere — runs, the interactive menu,
and `--leaderboard` (where a previously-scored model that has been dropped out
shows as `gone`). Add more folder names to `exclude_dirs` if you want additional
holding areas.

**Interactive menu (TUI).** `python main.py --tui` first shows the Archmage's
Roster: every model with its last score and last-run date. You then choose:

- `A` / Enter — run all models
- `N` — run only **untested** models (no record in the history DB)
- `<` — run only models whose last total score was **below** a number you enter
- `>` — run only models whose last total score was **above** a number you enter

Models that have never run count as eligible under `<` (so first runs and
previously-errored models get picked up). The `N` option is the opposite of a
score filter — it runs exactly the models you just added that have never been
benchmarked (models that ran and errored are excluded, since they were tested).

**CLI equivalents** (also work without the TUI):

```
python main.py --leaderboard        # print models ranked by score from the DB
python main.py --filter-untested    # run only models never benchmarked
python main.py --filter-lt 8        # run models that last scored < 8
python main.py --filter-gt 8        # run models that last scored > 8
python main.py --ingest results/<timestamp>   # backfill a past run into bench.db
```

`--leaderboard` reads the history DB and prints every model ranked by its latest
total score, with an **On Disk** flag: `YES` if the `.gguf` is still in the models
directory, or `gone` if it was scored in a past run but has since been deleted.
Handy for spotting low performers to cull and confirming what's already removed.

## Tweaking and rerunning

Everything is config-driven:

- Switch backend (`server` GPU vs `python` CPU), set `server_bin`, context size,
  GPU layers, temperature, or max tokens in `config/bench.yaml`.
- Change the flag in `config/bench.yaml`.
- Add or edit test cases in `config/tests/*.yaml` — no code changes needed.
- Add a new Test 2 transformation by adding a function to
  `harness/transforms.py` and referencing it by name in the test YAML.

Each run lands in its own timestamped folder, so you can diff `report.md` or the
CSVs across runs to compare parameter changes.

## Adding Test 3 (MCP) later

`harness/scoring.py` uses a `SCORERS` registry keyed by test id. To add a new
test type: create `config/tests/test3_*.yaml`, add a scorer function, register
it, and list the test in `bench.yaml`.
