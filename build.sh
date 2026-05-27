#!/usr/bin/env bash
# Build + push the RunPod worker image. Usage:
#   ./build.sh <docker-hub-username>
# Then deploy the endpoint at runpod.io/console/serverless using that tag.

set -euo pipefail

USER="${1:?usage: ./build.sh <docker-hub-username>}"
TAG="${2:-latest}"
IMAGE="${USER}/wake-art-gen:${TAG}"

cd "$(dirname "$0")"

# The LoRA must be in ./loras/ for the Dockerfile to find it.
mkdir -p loras
cp ../models/loras/dark-ghibli-fairytales.safetensors loras/

echo "Building $IMAGE"
docker build -t "$IMAGE" .

echo "Pushing $IMAGE"
docker push "$IMAGE"

echo
echo "Image pushed. Deploy steps:"
echo "  1. https://runpod.io/console/serverless"
echo "  2. + New Endpoint → custom worker → use container image: $IMAGE"
echo "  3. GPU: A40 48GB or A100 80GB (see README)"
echo "  4. Container disk: 30 GB (Z-Image cache lands here on first cold start)"
echo "  5. Max workers: start at 5; scale to taste"
echo "  6. Copy the endpoint ID into art-gen/.env as RUNPOD_ENDPOINT_ID"
