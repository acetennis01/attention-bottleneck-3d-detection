# Evaluation

## KITTI validation protocol

- Dataset: KITTI 3D Object Detection validation split (3,769 samples)
- Metric: strict 3D AP40 (`0.50` IoU for Pedestrian/Cyclist, `0.70` for Car)
- Checkpoint selection: overall 3D AP40 moderate
- Training: seed 0, 80 epochs, effective batch size 48, matched schedules

These are validation results, not official KITTI test-server scores. AP40 is
an object-detection metric, not classification accuracy.

## Best-checkpoint results

| Model | Best epoch | Easy | Moderate | Hard |
| :--- | ---: | ---: | ---: | ---: |
| LiDAR-only PointPillars | 28 | 49.5614 | 38.6868 | 36.1066 |
| **Camera–LiDAR MBT** | **58** | **52.1218** | **41.1813** | **38.2311** |
| **MBT improvement** |  | **+2.5604** | **+2.4945** | **+2.1245** |

### Moderate AP40 by class

| Class | LiDAR-only | MBT | Difference |
| :--- | ---: | ---: | ---: |
| Pedestrian | 30.3730 | **32.3544** | **+1.9814** |
| Cyclist | **34.1321** | 33.2382 | **-0.8939** |
| Car | 51.5553 | **57.9512** | **+6.3959** |
| **Overall** | 38.6868 | **41.1813** | **+2.4945** |

MBT improves overall moderate AP40 by **2.4945 points** (**6.45% relative**).
The gain is driven primarily by Car detection. These are single-seed results.

## Reevaluation artifacts

- LiDAR-only: `work_dirs/pointpillars_lidar_control_eb48_stable_momentum_seed0/best_epoch28_eval/`
- MBT: `work_dirs/mbt_bev_eb48_stable_momentum_seed0/best_epoch58_eval/`

## Temporal evaluation

Full videos were rendered for KITTI Raw drives `0001`, `0005`, `0014`,
`0015`, `0035`, and `0051` at 10 FPS and a display threshold of `0.30`.

The first panel shows the full 360-degree raw LiDAR scan. Orange points are
the camera-FOV subset passed to both models; other points are reference-only.
Predictions are independent per frame with no tracking or temporal smoothing.
KITTI Raw tracklets are qualitative references and omit some objects. The
models and displayed benchmark classes are Car, Pedestrian, and Cyclist.

Videos are stored at:

```text
work_dirs/temporal_visualization_drive_<ID>/
2011_09_26_drive_<ID>_full360_lidar_vs_mbt.mp4
```
