"""
RunPod Serverless handler — Wake codex SDXL worker.

Two checkpoints supported via input.model:
  - "sdxl-dark-ghibli"  → SDXL base + Dark Ghibli Fairytales LoRA (baked in)
  - "ponyplex"          → PonyPlex Pony-XL checkpoint (lazy-loaded from CivitAI
                          on first request, cached to /workspace for the
                          lifetime of the worker)

Models are downloaded at first use rather than baked into the image, so the
container stays small (~3 GB compressed) and pushes/pulls quickly. Each worker
pays the cold-start cost once (~3–5 min for SDXL base, ~3 min for PonyPlex),
then stays warm.

Required env vars on the RunPod endpoint config:
  CIVITAI_TOKEN — auth token for downloading PonyPlex from civitai.com
"""
from __future__ import annotations

import base64
import io
import os
import time
import traceback
from pathlib import Path

import runpod
import torch

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

LORA_PATH = Path("/app/loras/dark-ghibli-fairytales.safetensors")
PONYPLEX_CACHE = Path("/workspace/checkpoints/ponyplex_v10.safetensors")
PONYPLEX_DOWNLOAD_URL = "https://civitai.com/api/download/models/436407"

# Per-process pipeline cache so a worker handling many requests doesn't reload.
_PIPELINE_CACHE: dict[str, object] = {}


def _read_lora_with_synthetic_alphas(path: Path) -> dict:
    """LoRAs that ship without alpha keys (Civitai's Dark Ghibli) need them
    synthesized so diffusers' converter completes. alpha == rank collapses
    the scale to 1.0, which is the convention these LoRAs assume."""
    from safetensors.torch import load_file
    state_dict = load_file(str(path))
    if any(k.endswith(".alpha") for k in state_dict):
        return state_dict
    A_SUFFIX = ".lora_A.weight"
    for key in list(state_dict.keys()):
        if not key.endswith(A_SUFFIX):
            continue
        base = key[: -len(A_SUFFIX)]
        if base + ".alpha" not in state_dict:
            state_dict[base + ".alpha"] = torch.tensor(float(state_dict[key].shape[0]))
    return state_dict


def _download_ponyplex():
    """Lazy-download PonyPlex from CivitAI on first request. Cached to
    /workspace/checkpoints/ which persists for the worker's lifetime."""
    if PONYPLEX_CACHE.exists() and PONYPLEX_CACHE.stat().st_size > 1_000_000_000:
        return  # already present
    PONYPLEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("CIVITAI_TOKEN", "")
    if not token:
        raise RuntimeError("CIVITAI_TOKEN env var not set on endpoint config")
    # CloudFlare WAF in front of civitai.com blocks Python clients (urllib +
    # requests) based on TLS fingerprint (JA3), regardless of User-Agent.
    # Bot detection is more aggressive on datacenter IP ranges like RunPod's.
    # curl works because its TLS handshake matches "real client" patterns.
    # Shell out to curl rather than fight the WAF in Python.
    sep = "&" if "?" in PONYPLEX_DOWNLOAD_URL else "?"
    auth_url = f"{PONYPLEX_DOWNLOAD_URL}{sep}token={token}"
    print(f"[handler] downloading PonyPlex from {PONYPLEX_DOWNLOAD_URL} (via curl) …", flush=True)
    t0 = time.perf_counter()
    import subprocess
    tmp = PONYPLEX_CACHE.with_suffix(".tmp")
    result = subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "--retry-delay", "5",
         "--max-time", "600", "-o", str(tmp), auth_url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (rc={result.returncode}): {result.stderr.strip()[:300]}")
    tmp.replace(PONYPLEX_CACHE)
    size_gb = PONYPLEX_CACHE.stat().st_size / (1 << 30)
    print(f"[handler] PonyPlex downloaded ({size_gb:.1f} GB) in {time.perf_counter() - t0:.1f}s", flush=True)


def _load_sdxl_dark_ghibli(lora_scale: float):
    """SDXL base + Dark Ghibli LoRA. The LoRA's "Studio Ghibli Dark
    Fairytale" trigger is in the prompt the caller composes."""
    from diffusers import StableDiffusionXLPipeline
    print("[handler] loading SDXL base + Dark Ghibli LoRA …", flush=True)
    t0 = time.perf_counter()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")
    if LORA_PATH.exists():
        try:
            state_dict = _read_lora_with_synthetic_alphas(LORA_PATH)
            pipe.load_lora_weights(state_dict)
            try:
                pipe.set_adapters(["default_0"], adapter_weights=[lora_scale])
            except Exception:
                pass
            print("[handler] Dark Ghibli LoRA loaded", flush=True)
        except Exception as e:
            print(f"[handler] LoRA load failed (continuing without): {e}", flush=True)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    print(f"[handler] SDXL ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return pipe


def _load_ponyplex():
    """PonyPlex single-file checkpoint. Pony-XL is SDXL-architecture, so we
    use from_single_file with the SDXL pipeline class."""
    from diffusers import StableDiffusionXLPipeline
    _download_ponyplex()
    print("[handler] loading PonyPlex …", flush=True)
    t0 = time.perf_counter()
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(PONYPLEX_CACHE),
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    print(f"[handler] PonyPlex ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return pipe


def get_pipeline(model: str, lora_scale: float):
    key = f"{model}::{lora_scale:.3f}" if model == "sdxl-dark-ghibli" else model
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]
    if model == "sdxl-dark-ghibli":
        pipe = _load_sdxl_dark_ghibli(lora_scale)
    elif model == "ponyplex":
        pipe = _load_ponyplex()
    else:
        raise ValueError(f"unknown model: {model!r}")
    _PIPELINE_CACHE[key] = pipe
    return pipe


def handler(event):
    """
    Input shape:
      {
        "prompt": str,
        "negative": str,
        "steps": int,           # default 30
        "guidance": float,      # default 7.0 (SDXL default; 5.0 for PonyPlex)
        "width": int, "height": int,
        "seed": int,
        "model": "sdxl-dark-ghibli" | "ponyplex",
        "lora_scale": float,    # default 0.9 (SDXL+DG only)
      }
    Output:
      {
        "image_b64": str (PNG),
        "seed": int,
        "elapsed_secs": float,
        "model": str,
        "width": int, "height": int,
      }
    """
    try:
        inp = event.get("input") or {}
        prompt = inp.get("prompt")
        if not prompt:
            return {"error": "missing required field: prompt"}

        model = inp.get("model", "sdxl-dark-ghibli")
        negative = inp.get("negative") or None
        steps = int(inp.get("steps", 30))
        guidance = float(inp.get("guidance", 7.0))
        width = int(inp.get("width", 1024))
        height = int(inp.get("height", 1024))
        seed = int(inp.get("seed", 0))
        lora_scale = float(inp.get("lora_scale", 0.9))

        # SDXL prefers multiples of 64; both checkpoints use SDXL architecture.
        width = max(384, min(2048, (width // 64) * 64))
        height = max(384, min(2048, (height // 64) * 64))

        pipe = get_pipeline(model, lora_scale)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        t0 = time.perf_counter()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )
        image = result.images[0]
        elapsed = time.perf_counter() - t0

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=False)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        del result, generator
        import gc; gc.collect()
        torch.cuda.empty_cache()

        return {
            "image_b64": b64,
            "seed": seed,
            "elapsed_secs": elapsed,
            "model": model,
            "width": width,
            "height": height,
        }
    except Exception as e:
        print(f"[handler] ERROR: {e}\n{traceback.format_exc()}", flush=True)
        return {"error": f"{type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})
