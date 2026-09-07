"""Model runners and performance metrics.

Two interchangeable runners are provided; both load a single GGUF model and
expose a chat() method returning generated text plus timing/throughput stats:

  - ServerRunner: launches a prebuilt Vulkan `llama-server.exe` and talks to it
    over its OpenAI-compatible HTTP API. This is the GPU path (AMD RX 9070 /
    RDNA4 on Windows) and needs no compiler or SDK. RECOMMENDED.

  - ModelRunner: uses the in-process `llama-cpp-python` bindings. CPU path,
    handy on machines without a GPU build.

Both share the same interface so the engine can use either transparently:
  __init__(model_path, llama_cfg, gen_cfg)
  .load() / .chat(system_prompt, user_message) -> ChatResult / .close()
  attributes: model_name, size_bytes, load_time_s
"""

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationStats:
    """Performance metrics captured for a single generation call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_time_s: float = 0.0
    tokens_per_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "wall_time_s": round(self.wall_time_s, 4),
            "tokens_per_s": round(self.tokens_per_s, 2),
        }


@dataclass
class ChatResult:
    text: str
    stats: GenerationStats = field(default_factory=GenerationStats)


class ModelRunner:
    """Thin wrapper around a llama.cpp model with metric collection."""

    def __init__(self, model_path: str, llama_cfg: dict, gen_cfg: dict):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.llama_cfg = llama_cfg
        self.gen_cfg = gen_cfg
        self._llm = None
        self.load_time_s: float = 0.0
        self.size_bytes: int = os.path.getsize(model_path)

    def load(self) -> None:
        # Imported lazily so the rest of the harness (and --dry-run) works
        # even without llama-cpp-python installed.
        from llama_cpp import Llama

        kwargs = {
            "model_path": self.model_path,
            "n_ctx": self.llama_cfg.get("n_ctx", 4096),
            "n_gpu_layers": self.llama_cfg.get("n_gpu_layers", 0),
            "seed": self.llama_cfg.get("seed", 42),
            "verbose": self.llama_cfg.get("verbose", False),
        }
        n_threads = self.llama_cfg.get("n_threads")
        if n_threads:
            kwargs["n_threads"] = n_threads

        t0 = time.perf_counter()
        self._llm = Llama(**kwargs)
        self.load_time_s = time.perf_counter() - t0

    def chat(self, system_prompt: str, user_message: str) -> ChatResult:
        """Run one system+user turn and return text plus performance stats."""
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # CTF semantics: every user submission is a brand-new chat session with
        # no running context. Clear the KV cache so nothing from a prior case
        # can influence this generation. We only ever send [system, user], so
        # this guarantees a fresh context every time.
        try:
            self._llm.reset()
        except Exception:
            # Older/newer builds may name this differently; ignore if absent.
            pass

        t0 = time.perf_counter()
        resp = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self.gen_cfg.get("max_tokens", 512),
            temperature=self.gen_cfg.get("temperature", 0.0),
            top_p=self.gen_cfg.get("top_p", 1.0),
            repeat_penalty=self.gen_cfg.get("repeat_penalty", 1.1),
        )
        wall = time.perf_counter() - t0

        text = resp["choices"][0]["message"]["content"] or ""
        usage = resp.get("usage", {}) or {}
        completion_tokens = usage.get("completion_tokens", 0)

        stats = GenerationStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", 0),
            wall_time_s=wall,
            tokens_per_s=(completion_tokens / wall) if wall > 0 else 0.0,
        )
        return ChatResult(text=text, stats=stats)

    def close(self) -> None:
        # Release the model handle so memory is freed before loading the next.
        self._llm = None


def _free_port() -> int:
    """Ask the OS for a free TCP port to bind llama-server to."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerRunner:
    """Runs a model via a prebuilt Vulkan `llama-server.exe` over HTTP.

    Spawns one server per model (offloading to the GPU), sends each case as an
    independent chat completion, and reads llama.cpp's own timing data for
    accurate tokens/s. Shuts the server down before the next model loads.

    Expected llama_cfg keys:
      server_bin   : path to llama-server.exe (required)
      n_ctx        : context window (default 4096)
      n_gpu_layers : layers to offload; 999/-1 = all (default 999)
      n_threads    : optional CPU threads
      load_timeout_s : seconds to wait for the server to become ready (default 180)
    """

    def __init__(self, model_path: str, llama_cfg: dict, gen_cfg: dict):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.llama_cfg = llama_cfg
        self.gen_cfg = gen_cfg
        self.size_bytes = os.path.getsize(model_path)
        self.load_time_s: float = 0.0

        self._proc: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._base_url: str = ""

    def load(self) -> None:
        import requests  # local import so --dry-run works without it

        server_bin = self.llama_cfg.get("server_bin")
        if not server_bin or not os.path.exists(server_bin):
            raise FileNotFoundError(
                f"llama-server binary not found: {server_bin!r}. "
                f"Set llama.server_bin in the config to the Vulkan release exe."
            )

        self._port = _free_port()
        self._base_url = f"http://127.0.0.1:{self._port}"

        # -1 in llama-cpp-python means "all layers"; the server uses a large
        # number to mean the same, so normalize.
        ngl = self.llama_cfg.get("n_gpu_layers", 999)
        if ngl is None or ngl < 0:
            ngl = 999

        # Use an absolute model path: the server runs with cwd set to its own
        # directory (so it finds its DLLs), which would break a relative path.
        abs_model = os.path.abspath(self.model_path)

        cmd = [
            server_bin,
            "--model", abs_model,
            "--n-gpu-layers", str(ngl),
            "--ctx-size", str(self.llama_cfg.get("n_ctx", 4096)),
            "--port", str(self._port),
            "--host", "127.0.0.1",
        ]
        n_threads = self.llama_cfg.get("n_threads")
        if n_threads:
            cmd += ["--threads", str(n_threads)]

        # Run the server from its own directory so it finds its ggml-*.dll files.
        cwd = os.path.dirname(os.path.abspath(server_bin))

        t0 = time.perf_counter()
        self._proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # Poll /health until the model is loaded and serving.
        timeout = self.llama_cfg.get("load_timeout_s", 180)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {self._proc.returncode}) "
                    f"loading {self.model_name}."
                )
            try:
                r = requests.get(f"{self._base_url}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            self.close()
            raise TimeoutError(
                f"llama-server did not become ready within {timeout}s for "
                f"{self.model_name}."
            )
        self.load_time_s = time.perf_counter() - t0

    def chat(self, system_prompt: str, user_message: str) -> ChatResult:
        import requests

        if self._proc is None:
            raise RuntimeError("Server not started. Call load() first.")

        # Each request is independent: the server holds no state between calls,
        # so sending only [system, user] gives a fresh context every time,
        # matching the CTF's per-submission session model.
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": self.gen_cfg.get("max_tokens", 512),
            "temperature": self.gen_cfg.get("temperature", 0.0),
            "top_p": self.gen_cfg.get("top_p", 1.0),
            "repeat_penalty": self.gen_cfg.get("repeat_penalty", 1.1),
            "cache_prompt": False,  # do not reuse KV across cases
        }

        t0 = time.perf_counter()
        resp = requests.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload, timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        wall = time.perf_counter() - t0

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {}) or {}
        completion_tokens = usage.get("completion_tokens", 0)

        # Prefer llama.cpp's own eval-speed measurement when present; it excludes
        # HTTP overhead and is the true generation throughput.
        timings = data.get("timings", {}) or {}
        tps = timings.get("predicted_per_second")
        if not tps:
            tps = (completion_tokens / wall) if wall > 0 else 0.0

        stats = GenerationStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", 0),
            wall_time_s=wall,
            tokens_per_s=tps,
        )
        return ChatResult(text=text, stats=stats)

    def close(self) -> None:
        # Terminate the server so the GPU/VRAM is freed before the next model.
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
