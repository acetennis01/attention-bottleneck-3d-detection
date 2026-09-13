#!/usr/bin/env bash
set -Eeuo pipefail

DATASET_DIR="${1:-/mnt/ml-data/nuscenes}"
MMDET_ROOT="${2:-/home/abhiramannaluru/mmdetection3d}"
PYTHON_BIN="/home/abhiramannaluru/miniconda3/envs/openmmlab/bin/python"
BASE_URL="https://motional-nuscenes.s3.amazonaws.com/public/v1.0"
STATE_DIR="${DATASET_DIR}/.setup_state"

ARCHIVES=(
  "v1.0-trainval_meta.tgz"
  "v1.0-trainval01_blobs.tgz"
  "v1.0-trainval02_blobs.tgz"
  "v1.0-trainval03_blobs.tgz"
  "v1.0-trainval04_blobs.tgz"
  "v1.0-trainval05_blobs.tgz"
  "v1.0-trainval06_blobs.tgz"
  "v1.0-trainval07_blobs.tgz"
  "v1.0-trainval08_blobs.tgz"
  "v1.0-trainval09_blobs.tgz"
  "v1.0-trainval10_blobs.tgz"
)

mkdir -p "${DATASET_DIR}" "${STATE_DIR}"

for archive in "${ARCHIVES[@]}"; do
  marker="${STATE_DIR}/${archive}.extracted"
  archive_path="${DATASET_DIR}/${archive}"
  url="${BASE_URL}/${archive}"

  if [[ -f "${marker}" ]]; then
    echo "[$(date --iso-8601=seconds)] Already extracted: ${archive}"
    continue
  fi

  expected_bytes="$(curl --fail --silent --show-error --location --head "${url}" \
    | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {gsub("\\r", "", $2); print $2}' \
    | tail -n 1)"

  if [[ -z "${expected_bytes}" ]]; then
    echo "Could not determine the size of ${archive}" >&2
    exit 1
  fi

  current_bytes=0
  if [[ -f "${archive_path}" ]]; then
    current_bytes="$(stat --format='%s' "${archive_path}")"
  fi

  if [[ "${current_bytes}" != "${expected_bytes}" ]]; then
    echo "[$(date --iso-8601=seconds)] Downloading ${archive} (${expected_bytes} bytes)"
    curl \
      --fail \
      --location \
      --continue-at - \
      --retry 20 \
      --retry-delay 10 \
      --retry-all-errors \
      --output "${archive_path}" \
      "${url}"
  fi

  actual_bytes="$(stat --format='%s' "${archive_path}")"
  if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
    echo "Size mismatch for ${archive}: expected ${expected_bytes}, got ${actual_bytes}" >&2
    exit 1
  fi

  echo "[$(date --iso-8601=seconds)] Extracting ${archive}"
  tar --extract --gzip --file "${archive_path}" --directory "${DATASET_DIR}"
  touch "${marker}"
  rm -- "${archive_path}"
  echo "[$(date --iso-8601=seconds)] Finished ${archive}"
  df --human-readable "${DATASET_DIR}"
done

echo "[$(date --iso-8601=seconds)] nuScenes trainval download and extraction complete"

if [[ ! -f "${STATE_DIR}/mmdet3d_preprocessing.complete" ]]; then
  echo "[$(date --iso-8601=seconds)] Generating MMDetection3D metadata"
  cd "${MMDET_ROOT}"
  export PYTHONPATH="${MMDET_ROOT}"
  "${PYTHON_BIN}" \
    "${MMDET_ROOT}/projects/myfusion/tools/prepare_nuscenes_trainval.py" \
    --root-path "${DATASET_DIR}" \
    --extra-tag nuscenes \
    --max-sweeps 10
  touch "${STATE_DIR}/mmdet3d_preprocessing.complete"
fi

echo "[$(date --iso-8601=seconds)] Full nuScenes trainval setup complete"
