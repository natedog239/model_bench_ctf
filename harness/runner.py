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
import shlex
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


class _HttpChatMixin:
    """Shared chat() implementation for any llama-server OpenAI-compatible HTTP
    endpoint. Subclasses must set self._base_url (e.g. "http://host:port") and,
    optionally, self._api_key for a reverse proxy that enforces auth.
    """

    _base_url: str = ""
    _api_key: Optional[str] = None
    gen_cfg: dict = {}

    def chat(self, system_prompt: str, user_message: str) -> ChatResult:
        import requests

        if not self._base_url:
            raise RuntimeError("No server endpoint. Call load() first.")

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

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        t0 = time.perf_counter()
        resp = requests.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload, headers=headers, timeout=600,
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


class ServerRunner(_HttpChatMixin):
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

        # --jinja activates the model's own chat template (from GGUF metadata),
        # required for correct tool/function-call parsing. Default on. An
        # optional chat_template_file overrides a model's bundled template.
        if self.llama_cfg.get("jinja", True):
            cmd.append("--jinja")
            tmpl = self.llama_cfg.get("chat_template_file")
            if tmpl:
                cmd += ["--chat-template-file", str(tmpl)]

        # Flash Attention + KV-cache quantization (q8_0 roughly halves KV memory
        # vs f16, enabling large contexts). llama.cpp needs Flash Attention for
        # a quantized KV cache, so enable it implicitly when a cache type is set.
        cache_k = self.llama_cfg.get("cache_type_k")
        cache_v = self.llama_cfg.get("cache_type_v")
        fa = self.llama_cfg.get("flash_attn")
        if fa is None and (cache_k or cache_v):
            fa = "on"
        if fa is not None:
            cmd += ["--flash-attn", str(fa)]
        if cache_k:
            cmd += ["--cache-type-k", str(cache_k)]
        if cache_v:
            cmd += ["--cache-type-v", str(cache_v)]

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


class RemoteServerRunner(_HttpChatMixin):
    """Benchmarks a model on a llama-server already running on another machine.

    Unlike ServerRunner, this does NOT start or stop any process: the harness
    has no shell on the remote host, so you must launch llama-server there
    yourself before the run, e.g.:

        llama-server --model <model>.gguf --n-gpu-layers 999 \\
                     --host 0.0.0.0 --port 8080

    One running server hosts exactly one model, so a remote run benchmarks
    whatever model that server currently has loaded. To test a different model,
    restart the remote server with a different --model and run again.

    Expected remote_cfg keys:
      base_url    : e.g. "http://REMOTE_HOST_IP:8080" (required)
      model_name  : label used in reports/leaderboard (default: derived from URL)
      size_bytes  : optional; reported model size on disk (default 0, unknown)
      api_key     : optional bearer token if a reverse proxy enforces auth
      load_timeout_s : seconds to wait for /health to be ok (default 180)
    """

    def __init__(self, remote_cfg: dict, gen_cfg: dict):
        base_url = (remote_cfg.get("base_url") or "").rstrip("/")
        if not base_url:
            raise ValueError(
                "remote.base_url is required for the 'remote' backend, e.g. "
                "http://REMOTE_HOST_IP:8080"
            )
        self._base_url = base_url
        self._api_key = remote_cfg.get("api_key")
        self.remote_cfg = remote_cfg
        self.gen_cfg = gen_cfg

        self.model_name = remote_cfg.get("model_name") or f"remote:{base_url}"
        # No local file to stat; size is unknown unless the user supplies it.
        self.size_bytes = int(remote_cfg.get("size_bytes", 0) or 0)
        # The harness didn't start the process, so there's no meaningful load
        # time to measure. Reported as 0.0 (treat as N/A in reports).
        self.load_time_s: float = 0.0

    def load(self) -> None:
        """Verify the remote server is up and serving before benchmarking.

        This only polls /health; it never launches anything. If the server
        isn't running yet, start it on the remote host and re-run.
        """
        import requests

        timeout = self.remote_cfg.get("load_timeout_s", 180)
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            try:
                r = requests.get(f"{self._base_url}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return
                last_err = f"status {r.status_code}"
            except Exception as e:
                last_err = str(e)
            time.sleep(0.5)
        raise TimeoutError(
            f"Remote llama-server at {self._base_url} was not healthy within "
            f"{timeout}s ({last_err}). Make sure it's running and reachable: "
            f"llama-server --model <model>.gguf --host 0.0.0.0 --port <port>"
        )

    def close(self) -> None:
        # Nothing to tear down: the remote process is owned by the remote host.
        pass


def _ssh_base_cmd(ssh_cfg: dict) -> list:
    """Build the `ssh ...` prefix used to run a command on the remote host.

    Uses the local OpenSSH client and relies on key-based auth (no password
    prompt), typically via a Host alias in ~/.ssh/config. BatchMode is forced
    so a missing key fails fast instead of hanging on an interactive prompt.
    """
    host = ssh_cfg.get("host")
    if not host:
        raise ValueError("ssh.host is required for the 'ssh' backend (e.g. 'my-ssh-host').")
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    port = ssh_cfg.get("port")
    if port:
        cmd += ["-p", str(port)]
    user = ssh_cfg.get("user")
    cmd.append(f"{user}@{host}" if user else host)
    return cmd


def ssh_list_models(ssh_cfg: dict) -> list:
    """Return the remote .gguf model file paths under ssh.models_dir.

    Filters out llama.cpp's bundled tokenizer vocab files (ggml-vocab-*.gguf),
    which are not benchmarkable models.
    """
    models_dir = ssh_cfg.get("models_dir")
    if not models_dir:
        raise ValueError("ssh.models_dir is required for the 'ssh' backend.")
    # -maxdepth keeps discovery predictable; adjust in config if you nest models.
    remote = (
        f"find {shlex.quote(models_dir)} -maxdepth 3 -name '*.gguf' "
        f"! -name 'ggml-vocab-*' -printf '%s\\t%p\\n' 2>/dev/null | sort"
    )
    out = subprocess.run(
        _ssh_base_cmd(ssh_cfg) + [remote],
        capture_output=True, text=True, timeout=60,
    )
    models = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        size_str, path = line.split("\t", 1)
        try:
            size = int(size_str)
        except ValueError:
            size = 0
        models.append({"path": path, "size_bytes": size})
    return models


class SshServerRunner(_HttpChatMixin):
    """Orchestrates a llama-server on a remote host over SSH, one model at a time.

    Mirrors ServerRunner's load/chat/close lifecycle, but the process lives on
    the remote box: load() SSH-launches llama-server (backgrounded, PID captured)
    for a single model, health-checks it over HTTP, chat() sends cases over HTTP,
    and close() SSH-kills the PID to free VRAM before the next model loads.

    Requires non-interactive (key-based) SSH — typically a Host alias in
    ~/.ssh/config so `ssh <host>` connects without a password prompt.

    Expected ssh_cfg keys:
      host          : ssh host or alias (required, e.g. 'my-ssh-host')
      user          : optional ssh user (omit if encoded in the Host alias)
      port          : optional ssh port (default 22)
      server_bin    : remote path to llama-server (required)
      models_dir    : remote directory holding .gguf models (required)
      http_host     : address the harness connects to for HTTP (default = host)
      http_port     : port llama-server listens on remotely (default 8080)
      device        : Vulkan device to pin, e.g. 'Vulkan1' (v620) or 'Vulkan0'
                      (3070). Default 'Vulkan1'. Set null to let llama.cpp pick.
      n_ctx         : context window (default 4096; sized to fit 32GB VRAM)
      n_gpu_layers  : layers to offload; 999 = all (default 999)
      fit           : pass --fit on as an OOM safety net (default True)
      jinja         : pass --jinja to use the model's chat template, required
                      for tool/function-call parsing (default True)
      chat_template_file : optional remote path overriding a model's bundled
                      chat template (default None = use the model's own)
      cache_type_k  : KV cache K data type, e.g. 'q8_0' to halve KV memory and
                      fit large contexts (default None = f16)
      cache_type_v  : KV cache V data type (default None = f16)
      flash_attn    : --flash-attn 'on'/'off'/'auto'; auto-enabled ('on') when
                      a cache type is set, since quantized KV requires it
      load_timeout_s: seconds to wait for /health after launch (default 300)
    """

    def __init__(self, model: dict, ssh_cfg: dict, gen_cfg: dict):
        self.ssh_cfg = ssh_cfg
        self.gen_cfg = gen_cfg

        self.model_path = model["path"]                 # remote path
        self.model_name = os.path.basename(self.model_path)
        self.size_bytes = int(model.get("size_bytes", 0) or 0)
        self.load_time_s: float = 0.0

        http_host = ssh_cfg.get("http_host") or ssh_cfg.get("host")
        self._http_port = int(ssh_cfg.get("http_port", 8080))
        self._base_url = f"http://{http_host}:{self._http_port}"
        self._api_key = ssh_cfg.get("api_key")
        self._remote_pid: Optional[str] = None

    def _remote_launch_cmd(self) -> str:
        """Build the remote shell command that starts llama-server and prints
        its PID. Runs from the server bin's directory so it finds its .so files.
        """
        c = self.ssh_cfg
        server_bin = c.get("server_bin")
        if not server_bin:
            raise ValueError("ssh.server_bin is required for the 'ssh' backend.")

        ngl = c.get("n_gpu_layers", 999)
        if ngl is None or ngl < 0:
            ngl = 999

        parts = [
            shlex.quote(server_bin),
            "--model", shlex.quote(self.model_path),
            "--n-gpu-layers", str(ngl),
            "--ctx-size", str(c.get("n_ctx", 4096)),
            "--host", "0.0.0.0",
            "--port", str(self._http_port),
        ]

        # Pin the GPU. Default to the v620 (Vulkan1); use split-mode none so it
        # never tries to spill onto the small 3070.
        device = c.get("device", "Vulkan1")
        if device:
            parts += ["--device", str(device), "--split-mode", "none"]

        # --fit on lets llama-server auto-trim unset args to fit device memory,
        # a safety net against OOM on the largest models.
        if c.get("fit", True):
            parts += ["--fit", "on"]

        # --jinja activates the model's own chat template (from GGUF metadata),
        # which is required for correct tool/function-call parsing. Default on.
        # An optional chat_template_file overrides a model's (sometimes broken)
        # bundled template; it is a path ON THE REMOTE HOST.
        if c.get("jinja", True):
            parts.append("--jinja")
            tmpl = c.get("chat_template_file")
            if tmpl:
                parts += ["--chat-template-file", shlex.quote(str(tmpl))]

        # Flash Attention + KV-cache quantization. Quantizing the KV cache
        # (e.g. q8_0) roughly halves KV memory vs f16, which is what makes large
        # contexts (32K+) fit on the v620. llama.cpp requires Flash Attention
        # for a quantized KV cache, so turn it on whenever a cache type is set.
        cache_k = c.get("cache_type_k")
        cache_v = c.get("cache_type_v")
        fa = c.get("flash_attn")
        if fa is None and (cache_k or cache_v):
            fa = "on"  # KV quant needs flash attention; enable it implicitly
        if fa is not None:
            parts += ["--flash-attn", str(fa)]
        if cache_k:
            parts += ["--cache-type-k", str(cache_k)]
        if cache_v:
            parts += ["--cache-type-v", str(cache_v)]

        n_threads = c.get("n_threads")
        if n_threads:
            parts += ["--threads", str(n_threads)]

        server_cmd = " ".join(parts)
        bin_dir = os.path.dirname(server_bin) or "."
        # Fully detach the server from the SSH channel, otherwise `ssh` blocks
        # until the backgrounded process closes stdout/stderr (i.e. never, while
        # it's serving). setsid + redirecting all three fds (stdin from
        # /dev/null, stdout/stderr to a log) lets the ssh command return
        # immediately. We echo the PID so close() can kill exactly this process.
        log = f"/tmp/llama-server-{self._http_port}.log"
        # The `{ ...; }` group is required: without it, `&` binds to the whole
        # `cd && setsid CMD` list and SSH waits on the backgrounded subshell's
        # inherited fds (a 30s+ hang). Grouping applies the redirection and the
        # background to just the server, so ssh returns immediately with the PID.
        return (
            f"cd {shlex.quote(bin_dir)} && "
            f"{{ setsid {server_cmd} < /dev/null > {log} 2>&1 & echo $!; }}"
        )

    def load(self) -> None:
        import requests

        # 1) Launch the server remotely and capture its PID.
        launch = subprocess.run(
            _ssh_base_cmd(self.ssh_cfg) + [self._remote_launch_cmd()],
            capture_output=True, text=True, timeout=30,
        )
        pid = launch.stdout.strip().splitlines()[-1] if launch.stdout.strip() else ""
        if not pid.isdigit():
            raise RuntimeError(
                f"Failed to launch remote llama-server for {self.model_name}: "
                f"{launch.stderr.strip() or launch.stdout.strip() or 'no PID returned'}"
            )
        self._remote_pid = pid

        # 2) Poll /health over HTTP until the model is loaded and serving.
        timeout = self.ssh_cfg.get("load_timeout_s", 300)
        deadline = time.time() + timeout
        last_err = None
        t0 = time.perf_counter()
        while time.time() < deadline:
            if not self._remote_alive():
                self._dump_remote_log_into_error()
            try:
                r = requests.get(f"{self._base_url}/health", timeout=3)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    self.load_time_s = time.perf_counter() - t0
                    return
                last_err = f"status {r.status_code}"
            except Exception as e:
                last_err = str(e)
            time.sleep(1.0)

        self.close()
        raise TimeoutError(
            f"Remote llama-server did not become ready within {timeout}s for "
            f"{self.model_name} ({last_err})."
        )

    def _remote_alive(self) -> bool:
        if not self._remote_pid:
            return False
        chk = subprocess.run(
            _ssh_base_cmd(self.ssh_cfg) + [f"kill -0 {self._remote_pid} 2>/dev/null && echo up"],
            capture_output=True, text=True, timeout=15,
        )
        return "up" in chk.stdout

    def _dump_remote_log_into_error(self) -> None:
        """If the server died during load, surface the tail of its remote log."""
        log = f"/tmp/llama-server-{self._http_port}.log"
        tail = subprocess.run(
            _ssh_base_cmd(self.ssh_cfg) + [f"tail -n 20 {shlex.quote(log)} 2>/dev/null"],
            capture_output=True, text=True, timeout=15,
        )
        self.close()
        raise RuntimeError(
            f"Remote llama-server for {self.model_name} exited during load. "
            f"Last log lines:\n{tail.stdout.strip()}"
        )

    def close(self) -> None:
        # Free VRAM before the next model loads. We must be thorough: a stray
        # server left holding the port (or the GPU) would break every
        # subsequent model in a serial run.
        #
        # Because the server is launched under `setsid`, its PID is also its
        # process-group ID, so `kill -- -<pid>` signals the whole group. We also
        # fall back to killing by the exact model path in case the recorded PID
        # is stale, then confirm the HTTP port is actually released.
        model_q = shlex.quote(self.model_path)
        parts = []
        if self._remote_pid:
            pid = self._remote_pid
            parts.append(f"kill -- -{pid} 2>/dev/null; kill {pid} 2>/dev/null")
        # Fallback: match the unique model path in the server's argv.
        parts.append(f"pkill -f {model_q} 2>/dev/null")
        parts.append("sleep 1")
        if self._remote_pid:
            pid = self._remote_pid
            parts.append(f"kill -9 -- -{pid} 2>/dev/null; kill -9 {pid} 2>/dev/null")
        parts.append(f"pkill -9 -f {model_q} 2>/dev/null")
        parts.append("true")
        try:
            subprocess.run(
                _ssh_base_cmd(self.ssh_cfg) + ["; ".join(parts)],
                capture_output=True, text=True, timeout=25,
            )
        except Exception:
            pass
        self._remote_pid = None
