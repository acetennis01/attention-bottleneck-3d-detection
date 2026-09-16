#!/usr/bin/env bash
set -Eeuo pipefail

# Resumable nuScenes v1.0-trainval setup for the remote MMDetection3D host.
#
# Usage:
#   bash tools/setup_nuscenes_trainval_safe.sh [DATASET_DIR] [MMDET_ROOT]
#
# Defaults:
#   DATASET_DIR=$HOME/data/nuscenes
#   MMDET_ROOT=$HOME/mmdetection3d
#
# Set PREPARE_METADATA=0 to download, extract, verify, and link the dataset
# without generating the MMDetection3D train/validation info files.

DATASET_DIR="${1:-${HOME}/data/nuscenes}"
MMDET_ROOT="${2:-${HOME}/mmdetection3d}"
PREPARE_METADATA="${PREPARE_METADATA:-1}"
PYTHON_BIN="${HOME}/miniconda3/envs/openmmlab/bin/python"
BASE_URL="https://motional-nuscenes.s3.amazonaws.com/public/v1.0"
DOWNLOAD_DIR="${DATASET_DIR}/.downloads"
STATE_DIR="${DATASET_DIR}/.setup_state"
DATA_LINK="${MMDET_ROOT}/data/nuscenes_full"
MIN_INITIAL_FREE_GIB=500

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

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

for command_name in aria2c curl df find gzip stat tar awk grep; do
  command -v "${command_name}" >/dev/null \
    || fail "Required command is missing: ${command_name}"
done

[[ -x "${PYTHON_BIN}" ]] \
  || fail "Python environment not found: ${PYTHON_BIN}"
[[ -d "${MMDET_ROOT}/mmdet3d" ]] \
  || fail "MMDetection3D checkout not found: ${MMDET_ROOT}"
[[ "${PREPARE_METADATA}" == "0" || "${PREPARE_METADATA}" == "1" ]] \
  || fail "PREPARE_METADATA must be 0 or 1"

mkdir -p "${DATASET_DIR}" "${DOWNLOAD_DIR}" "${STATE_DIR}"

completed_count="$(find "${STATE_DIR}" -maxdepth 1 -type f \
  -name '*.extracted' | wc -l)"
if (( completed_count == 0 )); then
  available_kib="$(df -Pk "${DATASET_DIR}" | awk 'NR == 2 {print $4}')"
  minimum_kib="$((MIN_INITIAL_FREE_GIB * 1024 * 1024))"
  if (( available_kib < minimum_kib )); then
    fail "At least ${MIN_INITIAL_FREE_GIB} GiB free is required before a new download"
  fi
fi

log "Dataset directory: ${DATASET_DIR}"
log "Available space: $(df -h "${DATASET_DIR}" | awk 'NR == 2 {print $4}')"

for archive in "${ARCHIVES[@]}"; do
  marker="${STATE_DIR}/${archive}.extracted"
  archive_path="${DOWNLOAD_DIR}/${archive}"
  control_path="${archive_path}.aria2"
  url="${BASE_URL}/${archive}"

  if [[ -f "${marker}" ]]; then
    log "Already extracted: ${archive}"
    # The marker is written only after tar exits successfully. Remove a stale
    # copy of this exact archive if a previous run stopped before cleanup.
    if [[ -f "${archive_path}" ]]; then
      rm -- "${archive_path}"
    fi
    if [[ -f "${control_path}" ]]; then
      rm -- "${control_path}"
    fi
    continue
  fi

  expected_bytes="$(
    curl --fail --silent --show-error --location --head "${url}" \
      | awk 'BEGIN {IGNORECASE=1}
             /^content-length:/ {gsub("\\r", "", $2); size=$2}
             END {print size}'
  )"
  [[ "${expected_bytes}" =~ ^[0-9]+$ ]] \
    || fail "Could not determine the expected size of ${archive}"

  if [[ -f "${archive_path}" ]]; then
    current_bytes="$(stat --format='%s' "${archive_path}")"
    if (( current_bytes > expected_bytes )); then
      fail "Existing ${archive_path} is larger than the official archive; refusing to overwrite it"
    fi
  fi

  log "Downloading ${archive} (${expected_bytes} bytes)"
  aria2c \
    --continue=true \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=1M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --dir="${DOWNLOAD_DIR}" \
    --out="${archive}" \
    "${url}"

  actual_bytes="$(stat --format='%s' "${archive_path}")"
  [[ "${actual_bytes}" == "${expected_bytes}" ]] \
    || fail "Size mismatch for ${archive}: expected ${expected_bytes}, got ${actual_bytes}"

  log "Testing gzip stream: ${archive}"
  gzip --test "${archive_path}"

  # Reject unexpected absolute paths or parent traversal before extraction.
  if ! tar --list --gzip --file "${archive_path}" \
      | awk '
          /^\// {bad=1}
          /(^|\/)\.\.($|\/)/ {bad=1}
          END {exit bad ? 1 : 0}
        '; then
    fail "Unsafe path detected in ${archive}"
  fi

  log "Extracting ${archive}"
  tar \
    --extract \
    --gzip \
    --no-same-owner \
    --no-same-permissions \
    --file "${archive_path}" \
    --directory "${DATASET_DIR}"

  touch "${marker}"
  rm -- "${archive_path}"
  [[ ! -f "${control_path}" ]] || rm -- "${control_path}"
  log "Finished ${archive}; free space: $(df -h "${DATASET_DIR}" | awk 'NR == 2 {print $4}')"
done

for required_path in \
  "${DATASET_DIR}/samples" \
  "${DATASET_DIR}/sweeps" \
  "${DATASET_DIR}/maps" \
  "${DATASET_DIR}/v1.0-trainval/scene.json" \
  "${DATASET_DIR}/v1.0-trainval/sample.json" \
  "${DATASET_DIR}/v1.0-trainval/sample_data.json"; do
  [[ -e "${required_path}" ]] \
    || fail "Expected extracted path is missing: ${required_path}"
done

log "Validating nuScenes scene, sample, and split counts"
"${PYTHON_BIN}" - "${DATASET_DIR}" <<'PY'
import sys
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

root = Path(sys.argv[1]).resolve()
nusc = NuScenes(version='v1.0-trainval', dataroot=str(root), verbose=False)
splits = create_splits_scenes()

assert len(nusc.scene) == 850, len(nusc.scene)
assert len(nusc.sample) == 34_149, len(nusc.sample)
assert len(splits['train']) == 700, len(splits['train'])
assert len(splits['val']) == 150, len(splits['val'])

train_names = set(splits['train'])
val_names = set(splits['val'])
train_scene_tokens = {
    scene['token'] for scene in nusc.scene if scene['name'] in train_names
}
val_scene_tokens = {
    scene['token'] for scene in nusc.scene if scene['name'] in val_names
}
train_samples = sum(
    sample['scene_token'] in train_scene_tokens for sample in nusc.sample
)
val_samples = sum(
    sample['scene_token'] in val_scene_tokens for sample in nusc.sample
)
assert train_samples == 28_130, train_samples
assert val_samples == 6_019, val_samples

print('Verified: 850 scenes, 34,149 samples')
print('Verified: 700 train scenes / 28,130 train samples')
print('Verified: 150 val scenes / 6,019 val samples')
PY

mkdir -p "${MMDET_ROOT}/data"
if [[ -L "${DATA_LINK}" ]]; then
  [[ "$(readlink --canonicalize "${DATA_LINK}")" == \
      "$(readlink --canonicalize "${DATASET_DIR}")" ]] \
    || fail "${DATA_LINK} points somewhere else; refusing to replace it"
elif [[ -e "${DATA_LINK}" ]]; then
  fail "${DATA_LINK} already exists and is not a symlink"
else
  ln --symbolic "${DATASET_DIR}" "${DATA_LINK}"
fi
log "Dataset link ready: ${DATA_LINK} -> ${DATASET_DIR}"

if [[ "${PREPARE_METADATA}" == "0" ]]; then
  log "PREPARE_METADATA=0; stopping before metadata generation"
  exit 0
fi

mem_total_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
swap_total_kib="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
if (( mem_total_kib + swap_total_kib < 24 * 1024 * 1024 )); then
  fail "Metadata conversion requires at least 24 GiB of RAM plus swap"
fi
if (( swap_total_kib == 0 )); then
  log "Warning: no swap is configured; continuing because total RAM is sufficient"
fi

log "Generating train/validation metadata only (no ground-truth database)"
cd "${MMDET_ROOT}"
export PYTHONPATH="${MMDET_ROOT}"
"${PYTHON_BIN}" - "${DATASET_DIR}" <<'PY'
import os
import pickle
import sys
from pathlib import Path

from mmdet3d.utils import register_all_modules
from tools.dataset_converters import nuscenes_converter
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

register_all_modules(init_default_scope=True)

root = Path(sys.argv[1]).resolve()
root_path = str(root)
train_path = root / 'nuscenes_infos_train.pkl'
val_path = root / 'nuscenes_infos_val.pkl'
state_dir = root / '.setup_state'
state_dir.mkdir(exist_ok=True)

if train_path.is_file() != val_path.is_file():
    raise RuntimeError(
        'Only one train/validation info file exists; refusing to overwrite '
        'completed metadata. Inspect the dataset manually.'
    )

if not train_path.is_file():
    nuscenes_converter.create_nuscenes_infos(
        root_path,
        'nuscenes',
        version='v1.0-trainval',
        max_sweeps=10,
    )
else:
    print('Reusing existing train and validation info files.')

converter_cwd = root / '.mmdet3d_converter_cwd'
compatibility_link = converter_cwd / 'data' / 'nuscenes'
compatibility_link.parent.mkdir(parents=True, exist_ok=True)
if compatibility_link.is_symlink():
    if compatibility_link.resolve() != root:
        raise RuntimeError(f'{compatibility_link} points to the wrong dataset')
elif compatibility_link.exists():
    raise RuntimeError(f'{compatibility_link} exists and is not a symlink')
else:
    compatibility_link.symlink_to(root, target_is_directory=True)


def is_v2_info(path: Path) -> bool:
    with path.open('rb') as handle:
        info = pickle.load(handle)
    return (
        isinstance(info, dict)
        and 'metainfo' in info
        and 'data_list' in info
    )


original_cwd = Path.cwd()
try:
    os.chdir(converter_cwd)
    for split, info_path in (('train', train_path), ('val', val_path)):
        marker = state_dir / f'{split}_infos_v2.complete'
        if marker.is_file():
            print(f'Skipping completed {split} info upgrade.')
            continue
        if is_v2_info(info_path):
            print(f'Reusing already-converted {split} metadata.')
            marker.touch()
            continue
        update_pkl_infos(
            'nuscenes', out_dir=root_path, pkl_path=str(info_path)
        )
        marker.touch()
finally:
    os.chdir(original_cwd)

print(f'Train metadata: {train_path} ({train_path.stat().st_size} bytes)')
print(f'Val metadata:   {val_path} ({val_path.stat().st_size} bytes)')
PY

[[ -s "${DATASET_DIR}/nuscenes_infos_train.pkl" ]] \
  || fail "Train metadata was not created"
[[ -s "${DATASET_DIR}/nuscenes_infos_val.pkl" ]] \
  || fail "Validation metadata was not created"

if [[ -e "${DATASET_DIR}/nuscenes_gt_database" || \
      -e "${DATASET_DIR}/nuscenes_dbinfos_train.pkl" ]]; then
  log "Notice: a ground-truth database already exists; this script did not create it"
fi

log "Full nuScenes trainval setup is complete"
du -h \
  "${DATASET_DIR}/nuscenes_infos_train.pkl" \
  "${DATASET_DIR}/nuscenes_infos_val.pkl"
