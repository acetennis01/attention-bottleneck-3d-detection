# Matched camera ablations (2026-10-25)

**Completed:** all 30 cases passed. Published compact reports and tables are
under [results/camera_ablations](../results/camera_ablations/SUMMARY.md).
The launch/monitoring instructions below document the original run; no queue
needs resuming to obtain these scores.

Inference only; no retraining or changes to architectures, checkpoints or data.
All cases use the selected best checkpoint and its saved resolved training
config, strict state loading, and the complete validation split.

- KITTI: token sweep B1/B2/B4/B8/B16 and larger V2 B8/B16; 3,769 frames each.
- nuScenes: original four-token PointPillars MBT, eight-token multi-scale
  PointPillars MBT, and the new eight-token TransFusion-L/nuImages-Swin MBT;
  6,019 frames each. Failed pilots are not included.
- Conditions: correct images, black pixels before normalization, and seeded
  Sattolo-shuffled images (no fixed points). nuScenes shuffles all six camera
  slots together from another validation sample, retaining target calibration.
- No LiDAR masking is active during inference. Clean and corrupted nuScenes
  inputs are checked on a real sample for identical points and calibration.
- Runs are sequential, batch one. Predictions, full metrics, configs and logs
  are kept separately. No old evaluation directory is overwritten.
- Prelaunch checks: 17 corruption/pipeline tests passed and all ten model
  checkpoints loaded strictly into their intended architectures.

Remote output directory on Haddr's PC:
`/home/yab/mmdetection3d/work_dirs/camera_ablations_kitti_nuscenes_20261025/`.
The new TransFusion-L/Swin best-NDS checkpoint at iteration 110,000 runs first.

```bash
# Run these on Haddr's PC.
tail -f ~/mmdetection3d/work_dirs/camera_ablations_kitti_nuscenes_20261025_queue.log
watch -n 15 cat ~/mmdetection3d/work_dirs/camera_ablations_kitti_nuscenes_20261025/suite.json
```

Tmux session: `camera-ablations`. `suite.json` records active case, child PID,
checkpoint/config hashes and passed/failed cases. Per-case `attempt_XX.log`
contains frame-level progress. `comparison.json`, `overall_metrics.csv` and
`SUMMARY.md` are refreshed after each case. nuScenes full per-class/TP metrics
remain under each case's `results/pred_instances_3d/metrics_summary.json`;
KITTI reports retain all class metrics and formatted prediction pickles.

If interrupted, the suite's `--resume` option skips verified passed cases and
uses a new attempt directory for interrupted/failed cases. It does not recover
in-memory partial inference, silently overwrite predictions, or start training.

Correct-minus-corrupted differences are AP/percentage points. A drop supports
camera dependence under corruption, not statistical significance, robustness to
all camera failures, or attribution of the entire LiDAR-baseline gain to images.
The clean result must also reproduce the selected checkpoint's expected score.
