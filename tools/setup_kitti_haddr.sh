#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/yab/mmdetection3d
DATA_ROOT=/home/yab/data/kitti
DOWNLOAD_ROOT=/home/yab/data/kitti_downloads

timestamp() {
    date --iso-8601=seconds
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

mkdir -p "$DATA_ROOT" "$DOWNLOAD_ROOT" "$REPO/data"

if [[ -L "$REPO/data/kitti" ]]; then
    current_target=$(readlink -f "$REPO/data/kitti")
    if [[ "$current_target" != "$DATA_ROOT" ]]; then
        log "ERROR: $REPO/data/kitti points to $current_target"
        exit 1
    fi
elif [[ -e "$REPO/data/kitti" ]]; then
    log "ERROR: $REPO/data/kitti exists and is not a symlink"
    exit 1
else
    ln -s "$DATA_ROOT" "$REPO/data/kitti"
fi

archives=(
    data_object_image_2.zip
    data_object_velodyne.zip
    data_object_calib.zip
    data_object_label_2.zip
)

base_url=https://s3.eu-central-1.amazonaws.com/avg-kitti

for archive in "${archives[@]}"; do
    log "Downloading $archive with aria2c"
    aria2c \
        --continue=true \
        --max-connection-per-server=16 \
        --split=16 \
        --min-split-size=10M \
        --file-allocation=none \
        --dir="$DOWNLOAD_ROOT" \
        "$base_url/$archive"
    log "Testing $archive"
    unzip -tq "$DOWNLOAD_ROOT/$archive"
done

for archive in "${archives[@]}"; do
    log "Extracting $archive"
    unzip -q -o "$DOWNLOAD_ROOT/$archive" -d "$DATA_ROOT"
done

declare -A expected_counts=(
    [training/image_2]=7481
    [training/velodyne]=7481
    [training/calib]=7481
    [training/label_2]=7481
    [testing/image_2]=7518
    [testing/velodyne]=7518
    [testing/calib]=7518
)

for relative_path in "${!expected_counts[@]}"; do
    actual=$(find "$DATA_ROOT/$relative_path" -maxdepth 1 -type f | wc -l)
    expected=${expected_counts[$relative_path]}
    log "$relative_path: $actual files (expected $expected)"
    if [[ "$actual" -ne "$expected" ]]; then
        log "ERROR: unexpected file count in $relative_path"
        exit 1
    fi
done

mkdir -p "$DATA_ROOT/ImageSets"
split_url=https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets
for split in train val trainval test; do
    log "Downloading ImageSets/$split.txt"
    aria2c \
        --continue=true \
        --allow-overwrite=true \
        --auto-file-renaming=false \
        --dir="$DATA_ROOT/ImageSets" \
        --out="$split.txt" \
        "$split_url/$split.txt"
done

declare -A expected_splits=(
    [train]=3712
    [val]=3769
    [trainval]=7481
    [test]=7518
)

for split in "${!expected_splits[@]}"; do
    # The upstream split files do not consistently end with a newline, so
    # count non-empty records rather than newline characters.
    actual=$(awk 'NF { count += 1 } END { print count + 0 }' \
        "$DATA_ROOT/ImageSets/$split.txt")
    expected=${expected_splits[$split]}
    log "ImageSets/$split.txt: $actual entries (expected $expected)"
    if [[ "$actual" -ne "$expected" ]]; then
        log "ERROR: unexpected split size for $split"
        exit 1
    fi
done

source /home/yab/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab
export PYTHONPATH="$REPO"

cd "$REPO"
log "Starting MMDetection3D KITTI preprocessing"
python -u tools/create_data.py kitti \
    --root-path ./data/kitti \
    --out-dir ./data/kitti \
    --extra-tag kitti

touch "$DATA_ROOT/.setup_complete"
log "KITTI download, extraction, verification, and preprocessing completed"
