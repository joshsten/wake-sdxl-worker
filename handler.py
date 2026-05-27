"""
RunPod Serverless handler — Wake codex SDXL worker.

Two checkpoints supported via input.model:
  - "sdxl-dark-ghibli"  → SDXL base + Dark Ghibli Fairytales LoRA (baked in)
  - "juggernaut-xl"     → Juggernaut-XL v9 from RunDiffusion (HuggingFace
                          from_pretrained on first request, cached to
                          /workspace via HF_HOME for the worker's lifetime)

Models load at first use rather than at image build, so the container stays
small (~3 GB compressed). Each worker pays the cold-start cost once (~3–5 min
for SDXL base, ~3 min for Juggernaut), then stays warm.
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
HBO_LORA_CACHE = Path("/workspace/checkpoints/happy-bright-odd-illustrious.safetensors")
HBO_LORA_DOWNLOAD_URL = "https://civitai.com/api/download/models/2405821"

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
    /workspace/checkpoints/ which persists for the worker's lifetime.

    CloudFlare WAF in front of civitai.com fingerprints Python TLS clients
    (urllib + requests both) and 403s them on datacenter IP ranges. curl
    uses a TLS profile CloudFlare permits, so shell out rather than fight
    the WAF in-process.
    """
    if PONYPLEX_CACHE.exists() and PONYPLEX_CACHE.stat().st_size > 1_000_000_000:
        return
    PONYPLEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("CIVITAI_TOKEN", "")
    if not token:
        raise RuntimeError("CIVITAI_TOKEN env var not set on endpoint config")
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
    """PonyPlex: a fine-tune of Pony Diffusion v6 (CivitAI baseModel=Pony).

    The actual missing piece from prior attempts wasn't text-encoder layout or
    VAE precision — it was CLIP-skip. Pony Diffusion was trained with CLIP-
    skip 2 (use the penultimate text-encoder layer's hidden state, not the
    final one). Diffusers defaults to CLIP-skip 1, so the UNet was being fed
    embeddings from the wrong layer and produced noise regardless of prompt.

    The clip_skip=2 setting belongs on the pipeline __call__ (not the loader)
    — see the handler's inference call. The loader just needs to NOT inject
    SDXL-base text encoders, so Pony's bundled (booru-trained) ones get used.
    The fp16-fix VAE remains; SDXL latents from any Pony-derived UNet still
    overflow under the default VAE in fp16.
    """
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    _download_ponyplex()
    print("[handler] loading PonyPlex (bundled Pony text encoders + fp16-fix VAE) …", flush=True)
    t0 = time.perf_counter()
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(PONYPLEX_CACHE),
        torch_dtype=torch.float16,
        use_safetensors=True,
        vae=vae,
    )
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    print(f"[handler] PonyPlex ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return pipe


def _download_hbo_lora():
    """Lazy-download the Happy Bright Odd LoRA (Illustrious variant) from
    CivitAI. Same curl-via-subprocess pattern as PonyPlex — required because
    civitai's CloudFlare WAF 403s Python TLS fingerprints on datacenter IPs."""
    if HBO_LORA_CACHE.exists() and HBO_LORA_CACHE.stat().st_size > 50_000_000:
        return
    HBO_LORA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("CIVITAI_TOKEN", "")
    if not token:
        raise RuntimeError("CIVITAI_TOKEN env var not set on endpoint config")
    sep = "&" if "?" in HBO_LORA_DOWNLOAD_URL else "?"
    auth_url = f"{HBO_LORA_DOWNLOAD_URL}{sep}token={token}"
    print(f"[handler] downloading Happy-Bright-Odd LoRA …", flush=True)
    t0 = time.perf_counter()
    import subprocess
    tmp = HBO_LORA_CACHE.with_suffix(".tmp")
    result = subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "--retry-delay", "5",
         "--max-time", "300", "-o", str(tmp), auth_url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (rc={result.returncode}): {result.stderr.strip()[:300]}")
    tmp.replace(HBO_LORA_CACHE)
    size_mb = HBO_LORA_CACHE.stat().st_size / (1 << 20)
    print(f"[handler] HBO LoRA downloaded ({size_mb:.0f} MB) in {time.perf_counter() - t0:.1f}s", flush=True)


def _load_illustrious_hbo():
    """Illustrious-XL base + Happy Bright Odd LoRA — "whimsical bright" register.

    Illustrious is an SDXL-architecture anime fine-tune. The HBO LoRA was
    trained against the Illustrious-v0 base. Like Pony, Illustrious models are
    typically used with CLIP-skip 2; handler.py applies that automatically
    when model == "illustrious-hbo". Caller is expected to lead prompts with
    the LoRA's activation phrase: BrightHappyOddDaal.
    """
    from diffusers import StableDiffusionXLPipeline
    _download_hbo_lora()
    print("[handler] loading Illustrious-XL base + Happy-Bright-Odd LoRA …", flush=True)
    t0 = time.perf_counter()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "OnomaAIResearch/Illustrious-xl-early-release-v0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")
    try:
        state_dict = _read_lora_with_synthetic_alphas(HBO_LORA_CACHE)
        pipe.load_lora_weights(state_dict)
        try:
            pipe.set_adapters(["default_0"], adapter_weights=[1.0])
        except Exception:
            pass
        print("[handler] Happy-Bright-Odd LoRA loaded", flush=True)
    except Exception as e:
        print(f"[handler] HBO LoRA load failed (continuing without): {e}", flush=True)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    print(f"[handler] Illustrious+HBO ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return pipe


def _load_juggernaut_xl():
    """Juggernaut-XL v9 from the RunDiffusion HuggingFace mirror.

    Replaces the PonyPlex path entirely: PonyPlex's CivitAI single-file
    checkpoint never round-tripped cleanly through diffusers' from_single_file
    (7 worker iterations of debugging — text encoder layout, VAE overflow,
    Pony-vs-SDXL text-encoder base, none of it produced coherent output).
    Juggernaut-XL ships a proper from_pretrained tree on HF with text_encoder,
    text_encoder_2, unet, vae, and tokenizers as separate folders, so the
    standard SDXL loader picks everything up natively — no schema guesswork.
    Painterly-realistic register, contrasts cleanly against Dark Ghibli.
    """
    from diffusers import StableDiffusionXLPipeline
    print("[handler] loading Juggernaut-XL v9 from HF …", flush=True)
    t0 = time.perf_counter()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v9",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    print(f"[handler] Juggernaut-XL ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return pipe


def get_pipeline(model: str, lora_scale: float):
    key = f"{model}::{lora_scale:.3f}" if model == "sdxl-dark-ghibli" else model
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]
    if model == "sdxl-dark-ghibli":
        pipe = _load_sdxl_dark_ghibli(lora_scale)
    elif model == "juggernaut-xl":
        pipe = _load_juggernaut_xl()
    elif model == "ponyplex":
        pipe = _load_ponyplex()
    elif model == "illustrious-hbo":
        pipe = _load_illustrious_hbo()
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
        "guidance": float,      # default 7.0 (SDXL default; works for both models)
        "width": int, "height": int,
        "seed": int,
        "model": "sdxl-dark-ghibli" | "juggernaut-xl" | "ponyplex" | "illustrious-hbo",
        "clip_skip": int (optional),  # auto: 2 for ponyplex/illustrious-hbo, None otherwise
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
        # CLIP-skip: explicit input override wins, else default by model.
        # Pony Diffusion and its fine-tunes (PonyPlex) were trained with
        # clip_skip=2 — penultimate text-encoder layer. Diffusers default is 1.
        if "clip_skip" in inp:
            clip_skip = int(inp["clip_skip"])
        elif model in ("ponyplex", "illustrious-hbo"):
            # Pony + Illustrious lineages were both trained with CLIP-skip 2.
            clip_skip = 2
        else:
            clip_skip = None

        # SDXL prefers multiples of 64; all checkpoints use SDXL architecture.
        width = max(384, min(2048, (width // 64) * 64))
        height = max(384, min(2048, (height // 64) * 64))

        pipe = get_pipeline(model, lora_scale)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )
        if clip_skip is not None:
            call_kwargs["clip_skip"] = clip_skip

        t0 = time.perf_counter()
        result = pipe(**call_kwargs)
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
