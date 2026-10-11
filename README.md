# Remote llama.cpp Model Manager + CTF Bench

A generic manager for driving a remote (or local) [llama.cpp](https://github.com/ggml-org/llama.cpp)
GPU box: it lists the remote `*.gguf` library, loads one model onto a stable HTTP
endpoint, health-checks it, and lets you hot-swap to the next model on demand —
freeing VRAM between loads. Point any external harness at the fixed endpoint, run
your trials, then swap. Live stats probe the running server for its loaded context
window and real prefill/decode tok/s.

![Serve & hold — remote model-swap manager](manager.png)

```
python main.py --serve                # manual model-swap serve mode (no scoring)
python main.py --serve --backend ssh  # go straight to the remote GPU box
```

See [Serve mode](#serve-mode-manual-model-swap) below for the full command set.

---

## CTF Small-Model Benchmark Harness

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
cp config/bench.example.yaml config/bench.yaml   # then edit for your setup
```

Copy `config/bench.example.yaml` to `config/bench.yaml` and fill in your own
values (model paths, flag, and — for the `remote`/`ssh` backends — your host,
IP, and remote paths). `config/bench.yaml` is gitignored so your infrastructure
details stay out of version control; the example file is the tracked template.

GPU: download the llama.cpp Windows Vulkan release, extract it into the project,
and point `llama.server_bin` in `config/bench.yaml` at `llama-server.exe`
(`backend: server` is the default).

## Usage

```
python main.py                    # run all models (plain CLI)
python main.py --tui              # live Evil Wizard dashboard (with pre-run menu)
python main.py --backend ssh      # orchestrate a remote GPU box over SSH
python main.py --serve            # manual model-swap serve mode (no scoring)
python main.py --leaderboard      # print the leaderboard from the history DB
python main.py --filter-untested  # run only models never benchmarked
python main.py --filter-lt 8      # run models that last scored < 8
python main.py --filter-gt 8      # run models that last scored > 8
python main.py --report-only DIR  # regenerate a report from a past run
python main.py --ingest DIR       # index a past run into bench.db
```

**Backends.** `--backend` (or `backend:` in the config) selects where models
run: `server` (local GPU llama-server, default), `python` (local CPU), `remote`
(one llama-server you already started on another box), or `ssh` (the harness
starts/stops llama-server on a remote GPU box for you). With `--tui`, a pre-run
menu asks **local vs remote** first; choosing remote runs SSH discovery of the
remote model library, then the usual model-selection menu. See the `remote:` and
`ssh:` blocks in `config/bench.yaml`. The `ssh` backend needs key-based SSH
(e.g. a `Host` alias) and a `http_host` the harness can resolve by IP/DNS.

Everything is config-driven in `config/bench.yaml` (backend, GPU layers, flag,
generation params, `exclude_dirs`). Results land in `results/<timestamp>/`.

## Serve mode (manual model swap)

`python main.py --serve` turns the harness into a plain model swapper: it loads
**one** model, holds it on a stable HTTP endpoint, and lets you swap to another
model on demand. There is **no scoring and no test suite** here — this mode
exists for driving an **external** benchmark (e.g. an XSS harness container)
that owns its own orchestration and pass/fail judging. Point that harness at the
endpoint this mode prints, run your trials, then come back and load the next
model behind the same endpoint.

```
python main.py --serve                # choose backend interactively
python main.py --serve --backend ssh  # go straight to the remote GPU box
```

Models are launched with `--jinja` by default (set `jinja:` in the `ssh:`/`llama:`
config blocks), so each model's own chat template is used — this is required for
correct **tool / function-call** parsing. Point `chat_template_file:` at a
replacement template if a specific GGUF ships a broken one.

**Large contexts (KV-cache quantization).** KV cache at f16 is the main limit on
how long a context fits in VRAM. Quantizing it with `cache_type_k`/`cache_type_v:
"q8_0"` (in the `ssh:`/`llama:` blocks) roughly halves KV memory, which is what
lets 32K+ contexts fit on the 32GB v620. llama.cpp requires Flash Attention for a
quantized KV cache, so `flash_attn` is turned on automatically whenever a cache
type is set (override with `"on"`/`"off"`/`"auto"`). The budget is set by weights
+ KV: a ~7GB model with `q8_0` KV comfortably holds `n_ctx: 32768`; larger models
need a smaller `n_ctx` or they'll be trimmed by `--fit`. Set the cache types to
`null` to go back to full f16 KV. After loading, the serve-mode `[s]tats` command
reports the context window `/props` actually loaded, so you can confirm `--fit`
didn't silently trim it.

It reuses the same runners as the benchmark, so backends behave identically:

- **`ssh`** (recommended for the 32GB server) — lists the remote `*.gguf`
  library, launches llama-server for the chosen model on the box, health-checks
  it, and holds it. Swapping kills the current server to free VRAM before the
  next loads. Your harness targets the fixed `http_host:http_port` from the
  `ssh:` block the whole time; only the model behind it changes.
- **`server`** — spawns a local `llama-server.exe`. Note it binds to
  `127.0.0.1`, so a harness in a container or on another host can't reach it
  without a port forward / reverse proxy; prefer `ssh` for that case.
- **`remote`** — only health-checks a server you already started elsewhere. One
  server hosts one model, so there's nothing to swap (restart it with a
  different `--model` to change models).

While a model is held, serve mode waits at a prompt (ssh backend shown; other
backends offer a subset):

```
serve> [n]ext  [r]eload  [u]nload  [s]tats  [t]ail-log  [i]nfo  [q]uit
```

- **n** — pick a different model (unloads the current one first)
- **r** — reload the current model
- **u** — stop serving but keep the controller open
- **s** — live stats: probes the running server for its loaded **context
  window** (`/props`) and fires a tiny throwaway request to read llama.cpp's own
  `timings`, reporting **prefill** (prompt-eval tok/s) and **decode**
  (generation tok/s). Needs no server-side `--metrics` flag.
- **t** — tail the remote llama-server log over SSH (ssh backend only). Surfaces
  the per-request prompt-eval/eval timings your external harness is generating
  against the server (reads `/tmp/llama-server-<port>.log`).
- **i** — reprint the endpoint + model info
- **q** — unload and exit (always frees VRAM on the way out)

Note: the **s** probe sends one small request to the live server, so run it
between harness trials rather than mid-run if you want clean isolation. Your
harness's own responses already carry the same `timings`/`usage` fields
per request, so that's the most accurate per-trial source.

## Latest results

MODEL LEADERBOARD (from history DB)

| # | Model | Score | Avg tok/s | Last Run | On Disk |
| --- | --- | --- | --- | --- | --- |
| 1 | gemma-4-12B-it-qat-UD-Q4_K_XL.gguf | 15/17 | 66.64 | 2026-09-13 10:55:15 | YES |
| 2 | gemma-4-12b-it-Q6_K.gguf | 14/17 | 50.62 | 2026-09-13 10:55:15 | YES |
| 3 | gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf | 14/17 | 54.42 | 2026-09-13 11:42:44 | YES |
| 4 | gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf | 14/17 | 113.63 | 2026-09-13 10:55:15 | YES |
| 5 | Qwen3.8-27B-Q4_K_M.gguf | 14/17 | 18.70 | 2026-09-13 11:06:02 | gone |
| 6 | Swift-Qwen3.8-27B-Q6_K.gguf | 13/17 | 12.53 | 2026-09-13 11:06:02 | gone |
| 7 | gemma4-v2-Q6_K.gguf | 12/17 | 50.76 | 2026-09-13 10:55:15 | gone |
| 8 | Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf | 12/17 | 67.42 | 2026-09-13 11:06:02 | gone |
| 9 | gemma-4-E2B-it-UD-IQ2_M.gguf | 11/17 | 148.77 | 2026-09-07 00:33:08 | gone |
| 10 | gemma-4-E2B-it-UD-Q4_K_XL.gguf | 11/17 | 164.83 | 2026-09-07 01:35:15 | gone |
| 11 | gemma-4-E4B-it-Q6_K.gguf | 11/17 | 94.49 | 2026-09-13 10:55:15 | gone |
| 12 | Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf | 11/17 | 11.62 | 2026-09-13 11:06:02 | gone |
| 13 | NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf | 11/17 | 149.61 | 2026-09-07 01:35:15 | gone |
| 14 | Ornith-1.5-9B-Q4_K_M.gguf | 11/17 | 85.80 | 2026-09-13 10:55:15 | gone |
| 15 | Ornith-1.5-9B-Q8_0.gguf | 11/17 | 58.85 | 2026-09-13 10:55:15 | gone |
| 16 | flux2-klein-4b-uncensored-q6_k.gguf | 10/17 | 128.61 | 2026-09-07 01:35:15 | gone |
| 17 | Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf | 10/17 | 95.90 | 2026-09-07 01:35:15 | gone |
| 18 | qwen3.5-4B-super-coder.Q4_0.gguf | 10/17 | 128.51 | 2026-09-07 01:35:15 | gone |
| 19 | Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf | 10/17 | 85.52 | 2026-09-13 10:55:15 | gone |
| 20 | Nanbeige_Nanbeige4.2-3B-Q6_K_L.gguf | 9/17 | 80.75 | 2026-09-07 00:33:08 | gone |
| 21 | Qwen3.5-9B-IQ4_XS.gguf | 9/17 | 72.30 | 2026-09-07 00:33:08 | gone |
| 22 | Qwen3.6-27B-Q6_K.gguf | 9/17 | 10.82 | 2026-09-13 11:06:02 | gone |
| 23 | Qwythos-9B-Claude-Mythos-5-1M-MTP-Q8_0.gguf | 9/17 | 58.14 | 2026-09-07 01:35:15 | gone |
| 24 | Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf | 9/17 | 58.41 | 2026-09-07 01:35:15 | gone |
| 25 | Llama-3.2-1B-Instruct-Q8_0.gguf | 8/17 | 284.92 | 2026-09-07 00:33:08 | gone |
| 26 | gemma-3-4b-it-Q4_K_M.gguf | 7/17 | 142.90 | 2026-09-07 01:35:15 | gone |
| 27 | Spark-X2.5-4B.gguf | 7/17 | 63.53 | 2026-09-07 00:33:08 | gone |
| 28 | gemma-3-1B-it-QAT-Q4_0.gguf | 6/17 | 276.46 | 2026-09-07 00:33:08 | gone |
| 29 | K2-Horizon-1B-BF16.gguf | ERROR | - | 2026-09-07 00:13:23 | gone |
| 30 | Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-AMD-MTP-Q6_K.gguf | ERROR | - | 2026-09-13 11:06:02 | gone |
| 31 | Qwen3.8-27B-DFlash2-BF16.gguf | ERROR | - | 2026-09-07 00:13:23 | gone |
| 32 | Qwen3.8-27B-DFlash2-Q4_K_M.gguf | ERROR | - | 2026-09-07 00:13:23 | gone |
| 33 | Qwen3.8-27B-DFlash2-Q8_0.gguf | ERROR | - | 2026-09-07 00:13:23 | gone |

## GGUF Quantization & Model Flag Reference

| Category | Tag / Flag | 1-Sentence Explanation |
| --- | --- | --- |
| **Precision & Standard Quants** | **`BF16`** | Stores weights in unquantized 16-bit brain floating-point format to preserve original model quality without loss. |
|  | **`Q8_0`** | Applies straightforward 8-bit quantization across all tensors, serving as a virtually lossless baseline with minimal overhead. |
|  | **`Q6_K`** | Employs 6-bit k-quants across model layers to deliver output virtually indistinguishable from 16-bit precision. |
|  | **`Q6_K_L`** | Allocates 6-bit quantization with an even higher precision bias on key tensors than standard Q6_K to maximize fidelity. |
|  | **`Q4_0`** | Implements basic, uniform 4-bit block quantization across all model matrices for legacy compatibility. |
|  | **`Q4_K_M`** | Uses 4-bit k-quants with mixed precision on critical attention/feed-forward layers, serving as the standard sweet spot for quality and speed. |
| **Importance Matrix (I-Quants)** | **`IQ4_XS`** | Combines importance-matrix calibration with compact 4-bit packing for lower perplexity loss than legacy 4-bit quants at a small footprint. |
|  | **`IQ3_XXS`** | Uses an importance matrix to aggressively compress weights down to an ultra-compact ~3-bit footprint for extreme memory savings. |
|  | **`IQ2_M`** | Leverages an importance matrix to achieve usable 2-bit inference by protecting critical weights identified during calibration. |
| **Dynamic Quantization Modifiers** | **`UD-`** | Denotes Unsloth Dynamic quantization, which dynamically varies bit-depth layer-by-layer based on measured tensor sensitivity. |
|  | **`_XL`** | Indicates an extra-large dynamic quant variant that boosts key attention and feed-forward layers to higher bitwidths for near-tier-above quality. |
|  | **`KQuant-Dynamic`** | Combines k-quantization blocks with dynamic per-layer precision allocation to optimize trade-offs between perplexity and VRAM. |
| **Training & Fine-Tuning** | **`qat` / `QAT**` | Indicates Quantization-Aware Training was used during fine-tuning so the model learned to compensate for precision loss beforehand. |
|  | **`it` / `Instruct**` | Marks an instruction-tuned checkpoint optimized for conversational chat, system prompting, and task execution rather than raw text completion. |
|  | **`Fable-Fus` / `Fus**` | Designates an experimental composite merge or fusion combining narrative and reasoning capabilities from multiple donor checkpoints. |
|  | **`Uncensored` / `UnHeretic**` | Denotes an alignment ablation or uncensored fine-tune designed to remove built-in safety boundaries and refused prompts. |
| **Architectural Features & Acceleration** | **`MTP`** | Contains Multi-Token Prediction speculative drafting heads used to predict several tokens in parallel during inference. |
|  | **`DFlash2`** | Denotes integration or optimization with FlashAttention-2 / dynamic block-sparse attention mechanisms for high-throughput context handling. |
|  | **`A4B` / `A3B**` | Identifies the active parameter count in Mixture-of-Experts (MoE) architectures (e.g., 4B or 3B active weights per forward pass). |
|  | **`NM` / `DAU` / `AMD**` | Flags hardware or kernel-level vendor patches (such as Neural Magic kernels or AMD ROCm runtime adaptations) baked into the distribution. |