#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/mmdetection3d}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/openmmlab/bin/python}"
PROJECT="$ROOT/projects/myfusion"
OUTPUT="$ROOT/work_dirs/nuscenes_analysis"
MBT_CHECKPOINT="$ROOT/work_dirs/mbt_bev_nuscenes_fixed/best_NuScenes metric_pred_instances_3d_NuScenes_NDS_epoch_9.pth"
LIDAR_CHECKPOINT="$ROOT/work_dirs/pointpillars_nuscenes/best_NuScenes metric_pred_instances_3d_NuScenes_NDS_epoch_24.pth"

cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir -p "$OUTPUT"

run_evaluation() {
    local name="$1"
    local config="$2"
    local checkpoint="$3"
    local directory="$OUTPUT/$name"
    local summary="$directory/results/pred_instances_3d/metrics_summary.json"
    mkdir -p "$directory"
    if [[ -f "$summary" ]]; then
        echo "[$(date --iso-8601=seconds)] $name already complete"
        return
    fi
    echo "[$(date --iso-8601=seconds)] starting $name"
    "$PYTHON" -u tools/test.py \
        "$config" \
        "$checkpoint" \
        --work-dir "$directory" \
        --cfg-options \
            "test_evaluator.jsonfile_prefix=$directory/results" \
        >"$directory/test.log" 2>&1
    test -f "$summary"
    echo "[$(date --iso-8601=seconds)] finished $name"
}

run_evaluation \
    baseline \
    "$PROJECT/configs/pointpillars_lidar_control_nuscenes.py" \
    "$LIDAR_CHECKPOINT"
run_evaluation \
    correct \
    "$PROJECT/configs/my_fusion_mbt_bev_nuscenes.py" \
    "$MBT_CHECKPOINT"
run_evaluation \
    shuffled \
    "$PROJECT/configs/my_fusion_mbt_bev_nuscenes_shuffled_eval.py" \
    "$MBT_CHECKPOINT"
run_evaluation \
    black \
    "$PROJECT/configs/my_fusion_mbt_bev_nuscenes_black_eval.py" \
    "$MBT_CHECKPOINT"

"$PYTHON" "$PROJECT/tools/analyze_nuscenes_ablations.py" \
    --baseline "$OUTPUT/baseline/results/pred_instances_3d/metrics_summary.json" \
    --correct "$OUTPUT/correct/results/pred_instances_3d/metrics_summary.json" \
    --shuffled "$OUTPUT/shuffled/results/pred_instances_3d/metrics_summary.json" \
    --black "$OUTPUT/black/results/pred_instances_3d/metrics_summary.json" \
    --output-dir "$OUTPUT/report" \
    >"$OUTPUT/report_generation.log" 2>&1

echo "[$(date --iso-8601=seconds)] metrics analysis complete"
