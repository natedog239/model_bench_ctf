# CTF Small-Model Benchmark Harness

A lightweight, automated harness for finding the smallest local GGUF model that
can serve as the engine behind an AI-focused CTF. It runs each model through
[llama.cpp](https://github.com/ggml-org/llama.cpp) (prebuilt Windows Vulkan
release, GPU-accelerated on AMD), scores it on a set of CTF-style tests, records
performance, and ranks the results.

**Tests**
- **Auth gating** — the model holds a flag and must reveal it only when the user
  states they are authenticated (resists social engineering / prompt injection).
- **String manipulation** — the model must apply a requested string transform
  (reverse, base64, insert characters, etc.) to slip the flag past an external
  exact-match redaction engine.

Every case runs as a fresh, independent chat session (no context carried
between cases), matching the CTF's per-submission model.

## Latest results

MODEL LEADERBOARD (from history DB)

| # | Model | Score | Avg tok/s | Last Run | On Disk |
|---:|---|---:|---:|---|:---:|
| 1 | gemma-4-12b-it-Q6_K.gguf | 14/17 | 49.33 | 2026-09-07 01:35:15 | YES |
| 2 | gemma-4-12B-it-qat-UD-Q4_K_XL.gguf | 13/17 | 66.04 | 2026-09-07 01:35:15 | YES |
| 3 | gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf | 13/17 | 113.18 | 2026-09-07 01:35:15 | YES |
| 4 | gemma4-v2-Q6_K.gguf | 13/17 | 49.33 | 2026-09-07 01:35:15 | YES |
| 5 | gemma-4-E4B-it-Q6_K.gguf | 12/17 | 94.38 | 2026-09-07 01:35:15 | YES |
| 6 | Ornith-1.5-9B-Q4_K_M.gguf | 12/17 | 84.96 | 2026-09-07 01:35:15 | YES |
| 7 | Ornith-1.5-9B-Q8_0.gguf | 12/17 | 58.30 | 2026-09-07 01:35:15 | YES |
| 8 | Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf | 12/17 | 85.08 | 2026-09-07 01:35:15 | YES |
| 9 | gemma-4-E2B-it-UD-IQ2_M.gguf | 11/17 | 148.77 | 2026-09-07 00:33:08 | gone |
| 10 | gemma-4-E2B-it-UD-Q4_K_XL.gguf | 11/17 | 164.83 | 2026-09-07 01:35:15 | gone |
| 11 | NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf | 11/17 | 149.61 | 2026-09-07 01:35:15 | gone |
| 12 | flux2-klein-4b-uncensored-q6_k.gguf | 10/17 | 128.61 | 2026-09-07 01:35:15 | gone |
| 13 | Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf | 10/17 | 95.90 | 2026-09-07 01:35:15 | gone |
| 14 | qwen3.5-4B-super-coder.Q4_0.gguf | 10/17 | 128.51 | 2026-09-07 01:35:15 | gone |
| 15 | Nanbeige_Nanbeige4.2-3B-Q6_K_L.gguf | 9/17 | 80.75 | 2026-09-07 00:33:08 | gone |
| 16 | Qwen3.5-9B-IQ4_XS.gguf | 9/17 | 72.30 | 2026-09-07 00:33:08 | gone |
| 17 | Qwythos-9B-Claude-Mythos-5-1M-MTP-Q8_0.gguf | 9/17 | 58.14 | 2026-09-07 01:35:15 | YES |
| 18 | Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf | 9/17 | 58.41 | 2026-09-07 01:35:15 | YES |
| 19 | Llama-3.2-1B-Instruct-Q8_0.gguf | 8/17 | 284.92 | 2026-09-07 00:33:08 | gone |
| 20 | gemma-3-4b-it-Q4_K_M.gguf | 7/17 | 142.90 | 2026-09-07 01:35:15 | gone |
| 21 | Spark-X2.5-4B.gguf | 7/17 | 63.53 | 2026-09-07 00:33:08 | gone |
| 22 | gemma-3-1B-it-QAT-Q4_0.gguf | 6/17 | 276.46 | 2026-09-07 00:33:08 | gone |
| 23 | K2-Horizon-1B-BF16.gguf | ERROR | - | 2026-09-07 00:13:23 | gone |
| 24 | Qwen3.8-27B-DFlash2-BF16.gguf | ERROR | - | 2026-09-07 00:13:23 | gone |
| 25 | Qwen3.8-27B-DFlash2-Q4_K_M.gguf | ERROR | - | 2026-09-07 00:13:23 | gone |
| 26 | Qwen3.8-27B-DFlash2-Q8_0.gguf | ERROR | - | 2026-09-07 00:13:23 | gone |

## TUI demo

![TUI demo](tui.gif)

## Architecture

- **Runtime** — prebuilt Vulkan `llama-server.exe` (GPU on AMD RX 9070 / RDNA4,
  no compiler or SDK needed); optional CPU fallback via `llama-cpp-python`.
- **Engine** (`harness/engine.py`) — discovers models, runs one at a time
  (loaded, tested, unloaded), emits progress events, writes results.
- **Runners** (`harness/runner.py`) — `ServerRunner` (HTTP to llama-server) and
  `ModelRunner` (in-process CPU bindings); both capture tokens/s.
- **Tests** (`config/tests/*.yaml`) — data-driven cases; scored by
  `harness/scoring.py`. Add cases without touching code.
- **History** (`harness/history.py` → `bench.db`) — SQLite index of runs and
  per-model scores; powers the leaderboard and model-selection filters.
- **Reporting** (`harness/report.py`) — per-run `results.json`, `cases.csv`, and
  `report.md`.
- **TUI** (`harness/tui.py`) — an "Evil Wizard" live dashboard: a summoning
  landing, a spell-casting battle scene (the spell carries the running test),
  and a GAME OVER results screen.

`models/` is the immutable model library; nothing is moved or copied. Park
models you want to keep but not run in `models/dropout/`.

## Setup

```
./setup.ps1                     # venv + Python deps + CPU fallback wheel
.\.venv\Scripts\Activate.ps1
```

GPU: download the llama.cpp Windows Vulkan release, extract it into the project,
and point `llama.server_bin` in `config/bench.yaml` at `llama-server.exe`
(`backend: server` is the default).

## Usage

```
python main.py                    # run all models (plain CLI)
python main.py --tui              # live Evil Wizard dashboard (with pre-run menu)
python main.py --leaderboard      # print the leaderboard from the history DB
python main.py --filter-untested  # run only models never benchmarked
python main.py --filter-lt 8      # run models that last scored < 8
python main.py --filter-gt 8      # run models that last scored > 8
python main.py --report-only DIR  # regenerate a report from a past run
python main.py --ingest DIR       # index a past run into bench.db
```

Everything is config-driven in `config/bench.yaml` (backend, GPU layers, flag,
generation params, `exclude_dirs`). Results land in `results/<timestamp>/`.
