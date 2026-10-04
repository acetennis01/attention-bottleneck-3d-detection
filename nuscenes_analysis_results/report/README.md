# Corrected nuScenes MBT evaluation

All results use the official 6,019-sample nuScenes validation split and the best-NDS MBT checkpoint.

| Condition | NDS | mAP |
|---|---:|---:|
| LiDAR baseline | 0.4450 | 0.2561 |
| MBT: correct | 0.4529 | 0.2778 |
| MBT: shuffled | 0.4424 | 0.2559 |
| MBT: black | 0.4405 | 0.2466 |

## Correct-camera deltas

- vs lidar baseline: NDS +0.0079, mAP +0.0217
- vs shuffled cameras: NDS +0.0105, mAP +0.0219
- vs black cameras: NDS +0.0124, mAP +0.0312

## Training behavior

- Best validation epoch: 9.
- Camera aggregation remained normalized and the camera/fusion gradients remained nonzero throughout the recorded diagnostics.
- The saved best-NDS checkpoint, rather than the final epoch, is used for every camera-ablation evaluation.

A reduction under shuffled and black-camera evaluation demonstrates that the fused detector depends on scene-corresponding visual input. The LiDAR comparison measures the net benefit of the complete fusion system.

See `class_mean_ap.csv`, `overall_metrics.csv`, and the accompanying figures for detailed results.
