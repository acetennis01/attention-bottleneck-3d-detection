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

The MBT-configured detector improves overall moderate AP40 by **2.4945
points** (**6.45% relative**). The gain is driven primarily by Car detection.

## Camera-input ablation

The same epoch-58 MBT checkpoint was evaluated with correct images, all-black
images, and a deterministic seed-0 derangement in which every validation
LiDAR sample receives a different sample's image.

| Camera input | Easy | Moderate | Hard |
| :--- | ---: | ---: | ---: |
| Correct image | 52.1218 | 41.1813 | 38.2311 |
| Black image | 52.1218 | 41.1813 | 38.2311 |
| Shuffled image | 52.1216 | 41.1811 | 38.2311 |

Black and shuffled images leave AP40 effectively unchanged. Across 17,438
detections, mean absolute score differences from normal MBT are
`2.08e-8` (black) and `1.83e-8` (shuffled), with maxima below `3.89e-5`.
Therefore, the checkpoint is effectively invariant to image content. Its AP
gain over PointPillars cannot be attributed to meaningful camera fusion.

Prediction artifacts:

- Black: `work_dirs/mbt_bev_best_black_image_eval/pred_instances_3d.pkl`
- Shuffled: `work_dirs/mbt_bev_best_shuffled_image_eval/pred_instances_3d.pkl`

## Convergence

| Model | First AP40 | Best epoch | Best AP40 | Final AP40 |
| :--- | ---: | ---: | ---: | ---: |
| LiDAR-only | 19.0815 | 28 | 38.6868 | 36.5388 |
| MBT-configured | 19.2230 | 58 | 41.1813 | 38.8402 |

The complete validation history and plot are in `evaluation_assets/`.

## Qualitative temporal analysis

Full-360 temporal visualizations were generated for KITTI Raw drives `0001`,
`0005`, `0014`, `0015`, `0035`, and `0051`. Orange points are the
camera-field-of-view LiDAR subset passed to both models; gray points are
reference-only. Predictions are independent per frame, without tracking or
temporal smoothing. The midpoint contact sheet uses drives `0001`, `0015`,
and `0051`, selected before inspecting their predictions.

These videos are qualitative only: KITTI Raw tracklets are incomplete, and
untrained classes such as Truck and Van can appear without displayed labels.

## Interpretation and limitations

The results show that the MBT-configured architecture outperforms the matched
PointPillars baseline in this seed, but the camera ablation shows that the
improvement is not evidence of meaningful camera--LiDAR fusion. Possible
sources include the added LiDAR-side transformer, learned bottleneck priors,
and residual processing. The study currently uses one seed, validation rather
than official test-server results, no explicit camera-to-BEV alignment, and
incomplete Raw tracklets for qualitative figures.
