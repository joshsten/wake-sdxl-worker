#!/usr/bin/env bash
# Download Wake-specific CivitAI models that need an auth token.
# Skipped silently if CIVITAI_TOKEN isn't set; affected models just won't be
# available to workflows. HF-baked models (Juggernaut, Illustrious, SDXL,
# Z-Image, VAE, Dark Ghibli LoRA) are present unconditionally.
#
# Manifest format (one per line, pipe-separated):
#   slug|civitai-version-id|relative-path-under-/comfyui/models/|expected-size-mb
#
# Read from WAKE_CIVITAI_MODELS env var (set in the Dockerfile so the
# manifest lives next to the image build, not in a config file that drifts).
set -e

MODELS_DIR="/comfyui/models"

if [[ -z "${CIVITAI_TOKEN:-}" ]]; then
  echo "[wake-init] CIVITAI_TOKEN not set — skipping CivitAI model downloads." >&2
  exit 0
fi

if [[ -z "${WAKE_CIVITAI_MODELS:-}" ]]; then
  echo "[wake-init] WAKE_CIVITAI_MODELS manifest not set — nothing to fetch." >&2
  exit 0
fi

echo "[wake-init] checking CivitAI models …"

# WAKE_CIVITAI_MODELS uses \n literals because env vars don't carry newlines
# cleanly through Docker. Convert back to real newlines for iteration.
printf '%b\n' "${WAKE_CIVITAI_MODELS}" | while IFS='|' read -r slug version_id rel_path expected_mb; do
  [[ -z "${slug}" ]] && continue
  target="${MODELS_DIR}/${rel_path}"
  if [[ -f "${target}" ]]; then
    # Already present from a prior cold start — skip.
    actual_mb=$(($(stat -c%s "${target}") / 1048576))
    echo "[wake-init]   ${slug}: present (${actual_mb} MB) — skip"
    continue
  fi
  mkdir -p "$(dirname "${target}")"
  tmp="${target}.partial"
  url="https://civitai.com/api/download/models/${version_id}?token=${CIVITAI_TOKEN}"
  echo "[wake-init]   ${slug}: downloading v${version_id} → ${rel_path} (~${expected_mb} MB)"
  # curl, not wget, because CivitAI's CloudFlare WAF JA3-fingerprints Python
  # clients and wget on some bases gets matched too. curl gets through
  # consistently from datacenter IPs. --fail returns non-zero on HTTP errors.
  if ! curl -fsSL --retry 3 --retry-delay 5 --max-time 900 -o "${tmp}" "${url}"; then
    echo "[wake-init]   ${slug}: FAILED — model will be unavailable" >&2
    rm -f "${tmp}"
    continue
  fi
  mv "${tmp}" "${target}"
  actual_mb=$(($(stat -c%s "${target}") / 1048576))
  echo "[wake-init]   ${slug}: done (${actual_mb} MB)"
done

echo "[wake-init] CivitAI model check complete."
