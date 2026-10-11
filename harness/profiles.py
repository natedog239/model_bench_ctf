"""llama.cpp flag/profile system for the serve and benchmark backends.

The runners (ServerRunner, SshServerRunner) used to hardcode which llama-server
flags they knew about (n_ctx, n_gpu_layers, cache_type_k, ...). That meant any
other llama.cpp flag required a code change, and every model got identical
flags. This module replaces that with a data-driven layer:

  1. A canonical table (SPEC) mapping friendly config keys -> llama-server CLI
     flags, so the known knobs stay readable in YAML.

  2. `extra_args` -- an escape hatch on any backend block (or profile). Lets you
     pass ANY llama-server flag verbatim, so you never have to touch Python to
     use a flag the SPEC doesn't name. Accepts a few shapes (see _normalize_extra).

  3. `profiles:` -- a top-level map of named, reusable override bundles. A
     profile is a partial backend block (any SPEC key + extra_args) that gets
     merged OVER the base block. A profile may carry a `match:` glob (or list of
     globs) so it auto-applies to models whose filename matches, and a
     `backends:` list to restrict it to e.g. only ssh.

  4. `resolve(base_cfg, profiles, model_name, backend, selected)` -- merges the
     base block with the chosen/auto-matched profile and returns the effective
     settings dict. `build_cli(effective, ...)` turns that into the final argv.

Nothing here starts a process; it only builds argument lists, so it's trivially
testable and shared by every backend.
"""

import fnmatch
import shlex
from typing import Optional


# Friendly config key -> (llama-server flag, kind). `kind` controls rendering:
#   "value" : emit [flag, str(value)] when value is not None
#   "bool"  : emit [flag] when value is truthy (a presence flag like --mlock)
# Flags that need custom logic (flash-attn implied by KV quant, jinja +
# template, device + split-mode) are handled in build_cli, not here.
SPEC = {
    "n_ctx":        ("--ctx-size", "value"),
    "n_gpu_layers": ("--n-gpu-layers", "value"),
    "n_threads":    ("--threads", "value"),
    "n_threads_batch": ("--threads-batch", "value"),
    "batch_size":   ("--batch-size", "value"),
    "ubatch_size":  ("--ubatch-size", "value"),
    "n_parallel":   ("--parallel", "value"),
    "cache_type_k": ("--cache-type-k", "value"),
    "cache_type_v": ("--cache-type-v", "value"),
    "rope_scaling": ("--rope-scaling", "value"),
    "rope_scale":   ("--rope-scale", "value"),
    "rope_freq_base": ("--rope-freq-base", "value"),
    "rope_freq_scale": ("--rope-freq-scale", "value"),
    "yarn_orig_ctx": ("--yarn-orig-ctx", "value"),
    "tensor_split": ("--tensor-split", "value"),
    "main_gpu":     ("--main-gpu", "value"),
    "numa":         ("--numa", "value"),
    "load_mode":    ("--load-mode", "value"),
    "mlock":        ("--mlock", "bool"),
    "no_mmap":      ("--no-mmap", "bool"),
    "cont_batching": ("--cont-batching", "bool"),
    "no_kv_offload": ("--no-kv-offload", "bool"),
}

# Keys that are handled specially (not through SPEC) or are not llama-server
# flags at all (orchestration/transport settings). Kept here so resolve() can
# carry them through and callers can read them, but build_cli ignores them.
_NON_FLAG_KEYS = {
    "server_bin", "models_dir", "host", "user", "port", "http_host",
    "http_port", "api_key", "base_url", "model_name", "size_bytes",
    "load_timeout_s", "backend", "fit", "device", "jinja",
    "chat_template_file", "flash_attn", "split_mode", "extra_args",
    "profile", "profiles", "verbose", "seed",
}


def _normalize_extra(extra) -> list:
    """Turn an `extra_args` value into a flat list of CLI tokens.

    Accepts, in order of convenience:
      - a string:        "--mlock --parallel 4"   (shlex-split)
      - a list of str:   ["--mlock", "--parallel", "4"]
      - a list of pairs: [["--parallel", "4"], ["--mlock"]]
      - a dict:          {"--parallel": 4, "--mlock": true, "--n-cpu-moe": null}
                         (value true -> presence flag; null/false -> flag with
                          no value; anything else -> flag + str(value))

    A dict is the most readable for "a flag and its value", while a string is
    the fastest to paste from a shell. All shapes converge to a token list.
    """
    if extra is None:
        return []
    if isinstance(extra, str):
        return shlex.split(extra)
    tokens: list = []
    if isinstance(extra, dict):
        for k, v in extra.items():
            flag = str(k)
            if v is True or v is None:
                tokens.append(flag)
            elif v is False:
                continue  # explicit false disables a presence flag
            else:
                tokens += [flag, str(v)]
        return tokens
    if isinstance(extra, (list, tuple)):
        for item in extra:
            if isinstance(item, (list, tuple)):
                tokens += [str(x) for x in item]
            else:
                tokens.append(str(item))
        return tokens
    # Scalar (e.g. a lone number) -- stringify defensively.
    return [str(extra)]


def _merge_extra(base_extra, over_extra) -> list:
    """Concatenate two extra_args sources (base first, then override).

    We append rather than replace so a profile can ADD flags on top of the base
    block's extra_args. llama-server takes the last value for a repeated flag,
    so a profile naturally overrides a base value for the same flag.
    """
    return _normalize_extra(base_extra) + _normalize_extra(over_extra)


def match_profile(profiles: dict, model_name: str, backend: str,
                  selected: Optional[str] = None):
    """Return (profile_name, profile_dict) for a model, or (None, None).

    Resolution order:
      1. An explicitly selected profile name (if present and backend-eligible).
      2. The first profile whose `match:` glob(s) match the model filename and
         whose `backends:` (if set) include this backend. Insertion order in
         the YAML is honored, so put more specific profiles first.
    """
    if not profiles:
        return None, None

    def _eligible(p: dict) -> bool:
        backends = p.get("backends")
        return not backends or backend in backends

    if selected:
        p = profiles.get(selected)
        if p is None:
            raise KeyError(
                f"profile '{selected}' not found. Known profiles: "
                f"{', '.join(profiles) or '(none)'}"
            )
        if not _eligible(p):
            raise ValueError(
                f"profile '{selected}' is restricted to backends "
                f"{p.get('backends')}, not '{backend}'."
            )
        return selected, p

    for name, p in profiles.items():
        if not _eligible(p):
            continue
        pats = p.get("match")
        if not pats:
            continue
        if isinstance(pats, str):
            pats = [pats]
        if any(fnmatch.fnmatch(model_name, pat) for pat in pats):
            return name, p
    return None, None


def resolve(base_cfg: dict, profiles: Optional[dict], model_name: str,
            backend: str, selected: Optional[str] = None) -> dict:
    """Merge the base backend block with a matched/selected profile.

    Returns a new effective settings dict. Scalar keys from the profile replace
    the base; `extra_args` is CONCATENATED (base first) so a profile adds to,
    rather than wipes, the base escape-hatch flags. The chosen profile name is
    recorded under the private key "_profile_name" for display.
    """
    eff = dict(base_cfg or {})
    name, prof = match_profile(profiles or {}, model_name, backend, selected)
    if prof:
        base_extra = eff.get("extra_args")
        for k, v in prof.items():
            if k in ("match", "backends"):
                continue  # routing metadata, not a setting
            eff[k] = v
        eff["extra_args"] = _merge_extra(base_extra, prof.get("extra_args"))
    eff["_profile_name"] = name
    return eff


def _append_flag(cmd: list, key: str, value, quote):
    flag, kind = SPEC[key]
    if kind == "bool":
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    cmd += [flag, quote(str(value))]


def build_cli(server_bin: str, model_path: str, eff: dict, *,
              remote: bool, quote=None) -> list:
    """Build the full llama-server argv from effective settings.

    `remote=True` builds a command destined for a remote shell (ssh): it uses
    host 0.0.0.0, honors `device`/`split-mode`/`fit`, and shell-quotes paths.
    `remote=False` builds the local ServerRunner command (host 127.0.0.1); the
    caller supplies --port/--host itself for the local free-port case, so this
    returns everything EXCEPT host/port, which the caller appends.

    Precedence, lowest to highest:
      SPEC-mapped keys  ->  special-cased keys (flash-attn, jinja, device)  ->
      extra_args (appended last, so it wins for any repeated flag).

    `quote` is the quoting function for paths/values (shlex.quote for remote,
    identity for local argv lists). Defaults are chosen from `remote`.
    """
    if quote is None:
        quote = shlex.quote if remote else (lambda s: s)

    ngl = eff.get("n_gpu_layers", 999)
    if ngl is None or (isinstance(ngl, int) and ngl < 0):
        ngl = 999

    cmd = [server_bin, "--model", quote(model_path)]

    # SPEC-mapped simple flags. n_gpu_layers is normalized above, so inject the
    # cleaned value rather than the raw one.
    eff = dict(eff)
    eff["n_gpu_layers"] = ngl
    for key in SPEC:
        if key in eff:
            _append_flag(cmd, key, eff.get(key), quote)

    # Device pinning (remote multi-GPU). split-mode none keeps a big model off a
    # small secondary card. Local server usually has one device; still honored
    # if present.
    device = eff.get("device")
    if device:
        cmd += ["--device", quote(str(device))]
        split = eff.get("split_mode", "none")
        if split:
            cmd += ["--split-mode", quote(str(split))]

    # --fit safety net (remote/ssh default on; harmless elsewhere if set).
    if eff.get("fit"):
        cmd += ["--fit", "on"]

    # Flash Attention + KV-cache quant coupling: llama.cpp needs flash attention
    # for a quantized KV cache, so enable it implicitly when a cache type is set
    # and flash_attn wasn't given.
    cache_k = eff.get("cache_type_k")
    cache_v = eff.get("cache_type_v")
    fa = eff.get("flash_attn")
    if fa is None and (cache_k or cache_v):
        fa = "on"
    if fa is not None:
        cmd += ["--flash-attn", str(fa)]

    # Chat template: --jinja uses the model's own template (needed for tool
    # calls); an optional template file overrides a broken bundled one.
    if eff.get("jinja", True):
        cmd.append("--jinja")
        tmpl = eff.get("chat_template_file")
        if tmpl:
            cmd += ["--chat-template-file", quote(str(tmpl))]

    # Escape hatch LAST so it overrides anything above (llama-server takes the
    # last value for a repeated flag).
    cmd += _normalize_extra(eff.get("extra_args"))
    return cmd


def describe(eff: dict, server_bin: str, model_path: str, *, remote: bool,
             include_host_port: bool = True, http_port=None) -> list:
    """Return the resolved argv as a token list for DISPLAY (not execution).

    Mirrors build_cli but uses an identity quote so the user sees clean tokens,
    and optionally appends the host/port the runner would add, so the serve-mode
    [p]rofile view shows the exact command that will run.
    """
    cmd = build_cli(server_bin, model_path, eff, remote=remote,
                    quote=lambda s: s)
    if include_host_port:
        if remote:
            cmd += ["--host", "0.0.0.0", "--port", str(http_port or 8080)]
        else:
            cmd += ["--host", "127.0.0.1", "--port", "<auto>"]
    return cmd
