#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/yab/mmdetection3d
source /home/yab/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

smoke_root=${1:?Supply a new smoke output directory}
aggregation_mode=${2:-legacy}
case "$aggregation_mode" in
    legacy|valid_only) ;;
    *) printf 'Invalid aggregation mode: %s\n' "$aggregation_mode" >&2; exit 2 ;;
esac
mkdir -p "$smoke_root"
baseline='work_dirs/pointpillars_lidar_control_eb48_stable_momentum_seed0/best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_epoch_28.pth'
reference='work_dirs/mbt_bev_camera_focused_eb48_seed0/best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_epoch_20.pth'

for tokens in 1 2 4 8 16; do
    printf 'Starting KITTI B%s smoke at %s\n' "$tokens" "$(date --iso-8601=seconds)"
    python -u projects/myfusion/tools/smoke_kitti_bottlenecks.py \
        --baseline-checkpoint "$baseline" \
        --reference-checkpoint "$reference" \
        --aggregation-mode "$aggregation_mode" \
        --tokens "$tokens" --work-dir "$smoke_root/b$tokens" \
        > "$smoke_root/b$tokens.log" 2>&1
    printf 'KITTI B%s smoke passed\n' "$tokens"
done
python -u projects/myfusion/tools/smoke_kitti_bottlenecks.py \
    --baseline-checkpoint "$baseline" \
    --reference-checkpoint "$reference" \
    --aggregation-mode "$aggregation_mode" \
    --tokens 4 --mode reference --work-dir "$smoke_root/reference_b4" \
    > "$smoke_root/reference_b4.log" 2>&1
printf 'All KITTI smoke tests passed at %s\n' "$(date --iso-8601=seconds)"
