# Wake codex worker — ComfyUI backend, lean image.
#
# Extends runpod-workers/worker-comfyui (the maintained Comfy-on-RunPod image).
# Models — both HF-sourced and CivitAI-sourced — are downloaded by an init
# script at first container start, not baked into the image. This keeps the
# build under RunPod's 30-min build window (a prior attempt baking ~28 GB of
# checkpoints at build time hit the time cap during the registry push) and
# lets the same image serve endpoints that swap models in and out.
#
# Cold-start cost: ~5-15 min per worker the first time it sees a particular
# model, depending on which models the user actually asks for. Subsequent
# requests hit the warm /comfyui/models/ cache. RunPod's worker volumes
# survive between cold starts on the same physical worker.
#
# Recommended GPU: RTX A5000 24GB or RTX 4090 24GB.

ARG WORKER_COMFYUI_VERSION=5.8.5
FROM runpod/worker-comfyui:${WORKER_COMFYUI_VERSION}-base

WORKDIR /comfyui

# Pre-create the directory tree Comfy expects so the init script can drop
# files in without checking. RUN_INIT is single-threaded; no race here.
RUN mkdir -p models/checkpoints models/loras models/vae \
             models/diffusion_models models/text_encoders

# Dark Ghibli LoRA is small (~80 MB) and stable — bake it into the image so
# the SDXL+DG path doesn't need any runtime download.
COPY loras/dark-ghibli-fairytales.safetensors models/loras/dark-ghibli-fairytales.safetensors

COPY wake-init.sh /usr/local/bin/wake-init.sh
RUN chmod +x /usr/local/bin/wake-init.sh

# Belt-and-suspenders init hook: install at BOTH points where RunPod might
# invoke the worker — /start.sh (the Dockerfile CMD target) and an ENTRYPOINT
# we'll declare below. Whichever one fires first runs the init; the other
# becomes a no-op via the marker file.
#
# Prior attempt with just CMD ["/start.sh"] override never ran the init,
# suggesting RunPod's serverless framework dispatches through ENTRYPOINT or
# some custom invocation that skips CMD entirely.

# Replace /start.sh with a wrapper. Loud echos so worker logs show whether
# the hook fired.
RUN mv /start.sh /start.orig.sh && \
    printf '%s\n' \
        '#!/usr/bin/env bash' \
        'echo "[wake-start] /start.sh wrapper invoked"' \
        '/usr/local/bin/wake-init.sh || echo "[wake-start] init failed (rc=$?), continuing"' \
        'echo "[wake-start] exec /start.orig.sh"' \
        'exec /start.orig.sh "$@"' \
    > /start.sh && \
    chmod +x /start.sh

# Independent ENTRYPOINT wrapper — runs the init once via a marker, then
# delegates to whatever command argument follows.
RUN printf '%s\n' \
        '#!/usr/bin/env bash' \
        'echo "[wake-entry] ENTRYPOINT wrapper invoked"' \
        'if [[ ! -f /tmp/wake-init.done ]]; then' \
        '  /usr/local/bin/wake-init.sh && touch /tmp/wake-init.done' \
        'fi' \
        'echo "[wake-entry] exec $@"' \
        'exec "$@"' \
    > /usr/local/bin/wake-entrypoint.sh && \
    chmod +x /usr/local/bin/wake-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/wake-entrypoint.sh"]

# Manifest of remote models to fetch on first start, one per line:
#   src|name-or-version|relative-target|expected-size-mb
# src is "hf" for HuggingFace (URL is the name field) or "civitai" for
# CivitAI (name is the numeric version id; needs CIVITAI_TOKEN env var).
#
# Embedded in the env var so the manifest travels with the image; using a
# literal multiline file would survive layering more cleanly but the env
# form is easier to override per-endpoint (e.g., trim the manifest for a
# light deployment that only needs SDXL).
# Full wave-1 manifest — Z-Image included; total ~44 GB against the
# template's now-60 GB container disk.
ENV WAKE_MODEL_MANIFEST="\
hf|https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors|checkpoints/sdxl-base.safetensors|6800\n\
hf|https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors|checkpoints/juggernaut-xl-v9.safetensors|6800\n\
hf|https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors|checkpoints/illustrious-xl.safetensors|6800\n\
hf|https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors|vae/sdxl-vae-fp16-fix.safetensors|335\n\
hf|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors|text_encoders/qwen_3_4b.safetensors|4000\n\
hf|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors|diffusion_models/z_image_turbo_bf16.safetensors|12000\n\
hf|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors|vae/z-image-vae.safetensors|330\n\
civitai|436407|checkpoints/ponyplex-v1.safetensors|6500\n\
civitai|2405821|loras/happy-bright-odd-illustrious.safetensors|218\n\
civitai|2574210|loras/cursed-fairytale-sdxl.safetensors|218\n\
civitai|1578317|loras/digital-dystopia-illustrious.safetensors|109\
"

# Leave the parent's CMD ["/start.sh"] alone — our /start.sh is the wrapper
# now, so model init runs unconditionally before Comfy starts.
