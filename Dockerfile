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

# DreamShaper XL Turbo v2.1 — modern painted fantasy register (MTG / D&D 5e
# splash-art feel). Turbo variant: 6–8 steps at CFG 2, much faster inference
# than the SDXL base checkpoints, distilled for vivid composition.
RUN wget -q -O models/checkpoints/dreamshaper-xl-turbo-v2-1.safetensors \
        https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo/resolve/main/DreamShaperXL_Turbo_v2_1.safetensors

# Dark Ghibli (82 MB) baked from the repo — fits under GitHub's 100 MB
# single-file limit so it travels as a regular blob, no LFS gymnastics.
COPY loras/dark-ghibli-fairytales.safetensors models/loras/dark-ghibli-fairytales.safetensors

# Happy Bright Odd LoRA (218 MB) too big for a regular GitHub blob,
# RunPod CI can't pull git-lfs, and CivitAI's CloudFlare WAF rejects
# datacenter IPs (a build curl from there yielded an HTML challenge
# page that Comfy then tried to parse as safetensors → JSON decode
# error). Fix: host the file as a GitHub Release asset on this same
# repo. Release assets sit on release-assets.githubusercontent.com,
# which is a fast public CDN with no WAF interception. Verify the
# downloaded bytes look like a safetensors header so we catch a bad
# fetch at build time instead of at first inference.
# Use wget rather than curl — the worker-comfyui base image has wget
# preinstalled (it's what their own Dockerfile uses for model downloads)
# but not curl, which was the cause of every prior build's exit 127.
RUN wget -q --tries=3 --timeout=600 -O models/loras/happy-bright-odd-illustrious.safetensors "https://github.com/joshsten/wake-sdxl-worker/releases/download/wake-loras-v1/hbo-backup.safetensors" && ls -lh models/loras/happy-bright-odd-illustrious.safetensors

# ── ControlNet (structure guidance) ──────────────────────────────────────────
# SDXL Canny ControlNet (xinsir, ~2.5 GB) into /comfyui/models/controlnet/ so
# ComfyUI's stock ControlNetLoader can use it. The viewer's Image Forge extracts
# the Canny edge map LOCALLY and ships it as the control image, so NO
# comfyui_controlnet_aux preprocessor nodes are needed here — just the model.
# Filename must match workflows.py CONTROLNETS["canny"]. HF resolve URLs work
# from RunPod CI (unlike CivitAI's WAF); size-check guards against a bad fetch.
# Only Canny for now to stay well under the build-time/layer cap; depth/openpose/
# scribble can be added as further RUN layers (or via the wave-2 network volume)
# when needed.
RUN mkdir -p models/controlnet && \
    wget -q --tries=3 --timeout=900 -O models/controlnet/controlnet-canny-sdxl-1.0.safetensors \
        "https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors" && \
    [ "$(stat -c%s models/controlnet/controlnet-canny-sdxl-1.0.safetensors)" -gt 1000000000 ] && \
    ls -lh models/controlnet/controlnet-canny-sdxl-1.0.safetensors

# Parent's CMD (which serverless ignores anyway) and its handler.py stay as-is.
# No model-init hook needed: everything is already on disk.
