#!/usr/bin/env bash
# Download Wake-needed models at first container start. Idempotent: skips
# anything already present on /comfyui/models/. Logs each step to stdout so
# build-time investigation is straightforward.
#
# Manifest format (lines in $WAKE_MODEL_MANIFEST, pipe-separated):
#   src|locator|relative-target|expected-size-mb
# - src "hf"     : locator is the full HF resolve URL; no auth needed
# - src "civitai": locator is the numeric version id; requires CIVITAI_TOKEN
#
# HF downloads are parallel-safe (CDN); CivitAI downloads are sequential
# because the CDN tends to rate-limit aggressive bursts from the same IP.

set -u

MODELS_DIR="/comfyui/models"

if [[ -z "${WAKE_MODEL_MANIFEST:-}" ]]; then
  echo "[wake-init] WAKE_MODEL_MANIFEST not set — assuming the worker brings its own models." >&2
  exit 0
fi

echo "[wake-init] checking Wake model cache at ${MODELS_DIR} …"

# Convert \n literals (env vars can't carry real newlines through Docker) into
# real newlines for line-by-line iteration.
manifest=$(printf '%b' "${WAKE_MODEL_MANIFEST}")

# Split into HF + CivitAI groups so the HF ones can run in parallel.
hf_lines=()
civitai_lines=()
while IFS='|' read -r src locator rel expected_mb; do
  [[ -z "${src}" ]] && continue
  case "${src}" in
    hf)      hf_lines+=("${locator}|${rel}|${expected_mb}") ;;
    civitai) civitai_lines+=("${locator}|${rel}|${expected_mb}") ;;
    *)       echo "[wake-init] unknown src '${src}' — skipping" >&2 ;;
  esac
done <<< "${manifest}"

# Worker pool: fetch up to N HF files concurrently. The HF CDN handles burst
# parallelism well; serial would push cold-start time past 15 min.
hf_parallel=4
download_hf() {
  local url="$1" rel="$2" expected_mb="$3"
  local target="${MODELS_DIR}/${rel}"
  if [[ -f "${target}" ]]; then
    local actual_mb=$(($(stat -c%s "${target}" 2>/dev/null || echo 0) / 1048576))
    echo "[wake-init]   hf ${rel}: present (${actual_mb} MB) — skip"
    return 0
  fi
  mkdir -p "$(dirname "${target}")"
  local tmp="${target}.partial"
  local started=$(date +%s)
  if ! curl -fsSL --retry 3 --retry-delay 5 --max-time 900 -o "${tmp}" "${url}"; then
    echo "[wake-init]   hf ${rel}: FAILED (curl rc=$?)" >&2
    rm -f "${tmp}"
    return 1
  fi
  mv "${tmp}" "${target}"
  local actual_mb=$(($(stat -c%s "${target}") / 1048576))
  local elapsed=$(($(date +%s) - started))
  echo "[wake-init]   hf ${rel}: done (${actual_mb} MB in ${elapsed}s)"
}

# HF parallel batch
for line in "${hf_lines[@]}"; do
  IFS='|' read -r url rel expected_mb <<< "${line}"
  download_hf "${url}" "${rel}" "${expected_mb}" &
  # Throttle to hf_parallel concurrent downloads.
  while [[ $(jobs -rp | wc -l) -ge ${hf_parallel} ]]; do
    sleep 0.5
  done
done
wait

# CivitAI sequential — same curl pattern, with the token, and only if the
# token is set on the endpoint. If it's missing we silently skip the CivitAI
# models; the endpoint can still serve any HF-only workflow.
if [[ -n "${CIVITAI_TOKEN:-}" ]]; then
  for line in "${civitai_lines[@]}"; do
    IFS='|' read -r version_id rel expected_mb <<< "${line}"
    target="${MODELS_DIR}/${rel}"
    if [[ -f "${target}" ]]; then
      actual_mb=$(($(stat -c%s "${target}") / 1048576))
      echo "[wake-init]   civitai ${rel}: present (${actual_mb} MB) — skip"
      continue
    fi
    mkdir -p "$(dirname "${target}")"
    tmp="${target}.partial"
    started=$(date +%s)
    url="https://civitai.com/api/download/models/${version_id}?token=${CIVITAI_TOKEN}"
    if ! curl -fsSL --retry 3 --retry-delay 5 --max-time 900 -o "${tmp}" "${url}"; then
      echo "[wake-init]   civitai ${rel}: FAILED — model will be unavailable" >&2
      rm -f "${tmp}"
      continue
    fi
    mv "${tmp}" "${target}"
    actual_mb=$(($(stat -c%s "${target}") / 1048576))
    elapsed=$(($(date +%s) - started))
    echo "[wake-init]   civitai ${rel}: done (${actual_mb} MB in ${elapsed}s)"
  done
else
  echo "[wake-init] CIVITAI_TOKEN unset — skipping CivitAI models." >&2
fi

echo "[wake-init] model cache check complete."
exit 0
