# Wake codex worker — extends the official sdxl-prebaked Comfy worker.
#
# Architectural reset: RunPod's serverless invocation bypasses Docker
# CMD/ENTRYPOINT, so any runtime model-download hook is unreachable. The
# only reliable path is baking models into the image at build time. To
# stay under the 30-min build cap (a 28 GB single-layer push hit the cap),
# we start from runpod/worker-comfyui:<v>-sdxl which already has SDXL +
# fp16-fix VAE baked, and add only the Wake-specific deltas.
#
# Inherited from the parent (no need to download):
#   /comfyui/models/checkpoints/sd_xl_base_1.0.safetensors  (~6.8 GB)
#   /comfyui/models/vae/sdxl_vae.safetensors                (~330 MB)
#   /comfyui/models/vae/sdxl-vae-fp16-fix.safetensors       (~330 MB)
#
# Wave 1 deltas added here:
#   /comfyui/models/checkpoints/juggernaut-xl-v9.safetensors  (~6.8 GB)
#   /comfyui/models/checkpoints/illustrious-xl.safetensors    (~6.8 GB)
#   /comfyui/models/loras/dark-ghibli-fairytales.safetensors  (~80 MB, baked)
#
# Wave 1 ships three working styles: sdxl-base, sdxl-dark-ghibli, juggernaut-xl,
# and plain illustrious-xl. CivitAI models (PonyPlex / HBO / Cursed-Fairytale /
# Digital-Dystopia) and Z-Image-Turbo are deferred to a wave-2 network-volume
# integration — RunPod's network volumes survive across cold starts and can be
# pre-populated, sidestepping the build-cap problem entirely.

ARG WORKER_COMFYUI_VERSION=5.8.5
FROM runpod/worker-comfyui:${WORKER_COMFYUI_VERSION}-sdxl

WORKDIR /comfyui

# Each wget in its own RUN so they become separate layers — smaller per-layer
# pushes that fit under the cap, and Docker can resume individual layers if
# one push transiently fails. -q stays quiet; --show-progress=off if available
# (older wget versions only support -q).
RUN wget -q -O models/checkpoints/juggernaut-xl-v9.safetensors \
        https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors

RUN wget -q -O models/checkpoints/illustrious-xl.safetensors \
        https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors

# Dark Ghibli (82 MB) baked from the repo — fits under GitHub's 100 MB
# single-file limit so it travels as a regular blob, no LFS gymnastics.
COPY loras/dark-ghibli-fairytales.safetensors models/loras/dark-ghibli-fairytales.safetensors

# Happy Bright Odd LoRA (218 MB) is too big for a regular GitHub blob and
# RunPod's CI doesn't pull git-lfs objects (we tried — they end up as
# pointer text files that Comfy's LoraLoader fails to parse). Download it
# directly from CivitAI at build time with the token. The token has been
# shared through the build pipeline already and can be rotated server-side
# if compromised. ARG override lets us swap tokens without code edits.
ARG CIVITAI_TOKEN=b401f1141a02310cad754869322fa0d4
RUN curl -fsSL --retry 3 --retry-delay 5 --max-time 600 \
    -o models/loras/happy-bright-odd-illustrious.safetensors \
    "https://civitai.com/api/download/models/2405821?token=${CIVITAI_TOKEN}" && \
    ls -lh models/loras/happy-bright-odd-illustrious.safetensors

# Parent's CMD (which serverless ignores anyway) and its handler.py stay as-is.
# No model-init hook needed: everything is already on disk.
