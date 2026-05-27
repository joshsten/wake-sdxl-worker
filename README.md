# RunPod Serverless deploy — Wake art-gen

Deploy this worker once, then run `./batch.py --backend runpod` from your laptop and fan-out generation to RunPod's GPUs in parallel. Cuts a 24-hour balanced batch down to ~30 minutes on an A40, ~15 minutes on an A100.

## What's here

| File | Purpose |
|---|---|
| `Dockerfile` | CUDA 12.4 + Python 3.11 + diffusers + the Dark Ghibli LoRA, baked. Z-Image base model is downloaded on the first cold start and cached on the worker disk. |
| `handler.py` | RunPod handler. Module-level pipeline cache so each container keeps the model warm across many requests. |
| `requirements.txt` | Worker Python deps. |
| `build.sh` | Build + push to Docker Hub. |

## Prerequisites

- A RunPod account with billing set up (https://runpod.io)
- A Docker Hub account (or any container registry — adjust `build.sh` if elsewhere)
- `docker` installed locally
- ~10 GB upload bandwidth (the image is ~6 GB compressed)

## One-time deploy

```bash
cd art-gen/runpod_serverless

# 1. Build + push. Replace <username> with your Docker Hub username.
./build.sh <username>

# 2. Deploy at https://runpod.io/console/serverless
#    - New Endpoint → "Custom"
#    - Container Image: <username>/wake-art-gen:latest
#    - GPU types: A40 (cheapest reliable for Z-Image base) or A100 80GB
#    - Container Disk: 30 GB  (Z-Image cache lives here)
#    - Max Workers: 5–10 (fan-out width)
#    - Idle Timeout: 60s (workers spin down when batch finishes)
#    - Execution Timeout: 300s per job
#    - Active Workers: 0 (we don't want hot standby; pay only for use)

# 3. Copy the endpoint ID from the URL on the endpoint detail page
#    (e.g. https://www.runpod.io/console/serverless/user/endpoint/abc123 → abc123)

# 4. Set up the local client:
cd ..
cp .env.example .env
# Then edit .env with your RUNPOD_API_KEY (from runpod.io/console/user/settings)
# and RUNPOD_ENDPOINT_ID from step 3.
```

## Running

```bash
# Single test image
./art.py --backend runpod --slug elena-solara --preset quality

# Full batch, 5 concurrent workers
./batch.py --backend runpod --concurrent 5 --log outputs/runpod.jsonl

# Wider fan-out (RunPod scales workers automatically up to your "Max Workers" cap)
./batch.py --backend runpod --concurrent 20
```

## Expected timings (A40 48GB)

| Preset | Local 6750XT | RunPod A40 |
|---|---|---|
| fast (Turbo, 8 steps, 768²) | ~2 min/img | ~5 s/img |
| balanced (Turbo, 12 steps, 1024²) | ~6 min/img | ~12 s/img |
| quality (Base, 30 steps, 1024²) | ~14 min/img | ~25 s/img |
| max (Base, 40 steps, 1280²) | ~20 min/img | ~40 s/img |

For 220-slug balanced fan-out at concurrency=5: **~9 minutes** total (vs 22 hours local).

## Cost estimate

A40 PCIe spot rate on RunPod: ~$0.39/hr → ~$0.0001/s. For balanced (12s/img): ~$0.0013/img. **220 slugs ≈ $0.30.**

Quality preset (25s/img): ~$0.0027/img. 220 slugs ≈ $0.60.

If you bump to A100 80GB (~$1.19/hr, 2× faster on Z-Image): cost goes up modestly but throughput is much higher.

## Troubleshooting

- **First request takes 3–5 minutes:** that's the cold start — RunPod pulls the image, then the handler downloads Z-Image (~30 GB) into the worker's cache on first run. Subsequent requests on the same worker are fast.
- **`OOM` errors on the worker:** the chosen GPU is too small for the model + resolution. Bump to a larger GPU type (A40 48GB minimum for Z-Image base at 1024²).
- **`RUNPOD_ENDPOINT_ID not set`:** make sure `art-gen/.env` exists and is loaded by the client.
- **Image timing out (`TIMED_OUT`):** raise the Execution Timeout in the endpoint config (default 300s; quality preset on a slow worker can take 60–90s).
- **HTTP 401:** the API key is wrong; regenerate one at runpod.io/console/user/settings.
