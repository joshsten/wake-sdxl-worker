# Wake codex worker — ComfyUI backend.
#
# Strategy: extend runpod-workers/worker-comfyui (the official, maintained
# RunPod-on-ComfyUI image) and pre-stage the Wake-specific checkpoints + LoRAs
# at the layout Comfy expects. Diffusers' from_single_file path was unfixable
# for CivitAI checkpoints (10 worker iterations on PonyPlex / Illustrious never
# produced a coherent render) — Comfy's native loaders handle the same files
# without ceremony.
#
# Model placement (Comfy convention):
#   /comfyui/models/checkpoints/<name>.safetensors   — SDXL full checkpoints
#   /comfyui/models/loras/<name>.safetensors         — LoRAs
#   /comfyui/models/vae/<name>.safetensors           — VAEs
#   /comfyui/models/diffusion_models/<name>.safetensors  — Z-Image transformer
#   /comfyui/models/text_encoders/<name>.safetensors     — Z-Image text enc
#
# HF-sourced models bake at build time (no auth). CivitAI models lazy-download
# at first container start using the CIVITAI_TOKEN env var already configured
# on the endpoint — saves ~7 GB of secret-token-required downloads at build.

ARG WORKER_COMFYUI_VERSION=5.8.5

FROM runpod/worker-comfyui:${WORKER_COMFYUI_VERSION}-base AS base

WORKDIR /comfyui

RUN mkdir -p models/checkpoints models/loras models/vae models/diffusion_models models/text_encoders

# ─── Stage 1: HF-sourced checkpoints (no auth) ───────────────────────────────

# SDXL base 1.0 — universal anchor for the Dark Ghibli LoRA path and as a
# fallback comparison render.
RUN wget -q --show-progress=off -O models/checkpoints/sdxl-base.safetensors \
        https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# Juggernaut-XL v9 — painterly-realistic SDXL fine-tune (single-file from
# RunDiffusion's HF mirror; same weights as their CivitAI release).
RUN wget -q --show-progress=off -O models/checkpoints/juggernaut-xl-v9.safetensors \
        https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors

# Illustrious-XL v0.1 — anime/illustrative SDXL fine-tune, base for the
# Happy-Bright-Odd + Digital-Dystopia LoRAs.
RUN wget -q --show-progress=off -O models/checkpoints/illustrious-xl.safetensors \
        https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors

# fp16-fix VAE — required for any SDXL family checkpoint to decode under fp16
# without latent overflow. Loaded explicitly by every workflow.
RUN wget -q --show-progress=off -O models/vae/sdxl-vae-fp16-fix.safetensors \
        https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors

# Z-Image-Turbo from Comfy-Org's split-file mirror (matches the layout the
# official Z-Image Comfy workflow expects: transformer / text-encoder / VAE
# as separate files in separate folders).
RUN wget -q --show-progress=off -O models/text_encoders/qwen_3_4b.safetensors \
        https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors && \
    wget -q --show-progress=off -O models/diffusion_models/z_image_turbo_bf16.safetensors \
        https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors && \
    wget -q --show-progress=off -O models/vae/z-image-vae.safetensors \
        https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors

# ─── Stage 2: baked LoRAs (Dark Ghibli was already in the worker repo) ──────

COPY loras/dark-ghibli-fairytales.safetensors models/loras/dark-ghibli-fairytales.safetensors

# ─── Stage 3: CivitAI lazy-init script ──────────────────────────────────────
#
# CivitAI models (PonyPlex, Happy-Bright-Odd, Cursed-Fairytale, Digital-
# Dystopia) need an auth token. We download them on first container start
# rather than at build because (a) RunPod doesn't pass build-time secrets
# trivially and (b) lazy-download lets the same image run on endpoints that
# don't have the token configured (the worker just won't expose those models).
#
# The official worker-comfyui CMD is `/start.sh`. We wrap it: our script runs
# the CivitAI downloads, then exec's the original entrypoint.

COPY wake-civitai-init.sh /usr/local/bin/wake-civitai-init.sh
RUN chmod +x /usr/local/bin/wake-civitai-init.sh

ENV WAKE_CIVITAI_MODELS="\
ponyplex|436407|checkpoints/ponyplex-v1.safetensors|6500\n\
happy-bright-odd-il|2405821|loras/happy-bright-odd-illustrious.safetensors|218\n\
cursed-fairytale-sdxl|2574210|loras/cursed-fairytale-sdxl.safetensors|218\n\
digital-dystopia-il|1578317|loras/digital-dystopia-illustrious.safetensors|109\
"

# Override CMD: download CivitAI models, then hand off to the parent's start.sh.
CMD ["/bin/bash", "-c", "/usr/local/bin/wake-civitai-init.sh && exec /start.sh"]
