# Wake codex SDXL serverless worker (lean — models lazy-load at runtime).
#
# Image is ~3 GB compressed. First cold start downloads SDXL base (~7 GB)
# and Dark Ghibli LoRA from /app (baked). PonyPlex (~6.5 GB) lazy-downloads
# from CivitAI on first ponyplex request and caches to /workspace.
#
# Required endpoint env vars:
#   CIVITAI_TOKEN — for downloading PonyPlex
#
# Recommended GPU: RTX A5000 24GB or RTX 4090 24GB.

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# PyTorch built against CUDA 12.8 — Blackwell (sm_120) support landed in
# torch 2.7+ and only the cu128 wheel channel ships sm_120 kernels. RunPod's
# pool may allocate RTX PRO 6000 Blackwell MIG slices even when the endpoint
# requests A5000/L4/3090, so we need the broadest possible kernel coverage.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bake the LoRA (~80 MB).
COPY loras/dark-ghibli-fairytales.safetensors /app/loras/dark-ghibli-fairytales.safetensors

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
