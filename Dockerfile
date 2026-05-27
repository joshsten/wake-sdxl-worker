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

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

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

# PyTorch CUDA 12.1 wheels (~2 GB).
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bake the LoRA (~80 MB).
COPY loras/dark-ghibli-fairytales.safetensors /app/loras/dark-ghibli-fairytales.safetensors

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
