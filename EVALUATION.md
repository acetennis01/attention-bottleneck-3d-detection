# Evaluation

## Protocol

- KITTI 3D Object Detection validation split: 3,769 samples
- Classes: Car, Pedestrian, and Cyclist
- Metric: KITTI AP40
- Checkpoint selection: overall 3D AP40 at moderate difficulty
- Random seed: 0
- Effective batch size: 48

These are validation-split results, not official KITTI test-server scores.
AP40 is an object-detection average-precision metric, not classification
accuracy.

## Overall comparison

| Model | BEV easy | BEV moderate | BEV hard | 3D easy | 3D moderate | 3D hard |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| LiDAR-only PointPillars | 64.5245 | 52.8874 | 49.6791 | 49.5614 | 38.6868 | 36.1066 |
| **Attention-bottleneck fusion** | **66.3382** | **54.6478** | **51.9588** | **54.4341** | **42.6696** | **39.9782** |
| **Difference** | **+1.8137** | **+1.7604** | **+2.2797** | **+4.8727** | **+3.9828** | **+3.8716** |

The LiDAR baseline reached its best result at epoch 28. The camera-focused
fusion model reached its best result at epoch 20 of its 40-epoch run.

## Per-class moderate 3D AP40

| Class | LiDAR-only | Fusion | Difference |
| :--- | ---: | ---: | ---: |
| Pedestrian | 30.3730 | **32.2419** | **+1.8689** |
| Cyclist | 34.1321 | **37.7407** | **+3.6086** |
| Car | 51.5553 | **58.0263** | **+6.4710** |

## Camera-input ablation

The epoch-20 fusion checkpoint was evaluated without changing its LiDAR input
or annotations.

| Camera input | BEV AP40 moderate | 3D AP40 moderate |
| :--- | ---: | ---: |
| Correct image | **54.6478** | **42.6696** |
| Black image | 54.1208 | 42.3851 |
| Shuffled image | 53.9806 | 42.0954 |

Shuffling the camera image reduced moderate BEV AP40 by 0.6672 points and
moderate 3D AP40 by 0.5742 points. This demonstrates measurable use of camera
content, although LiDAR remains the dominant modality and one random seed is
not sufficient to establish statistical significance.

## Attention visualization

[`docs/attention_maps_camera_bev.png`](docs/attention_maps_camera_bev.png)
shows attention rollout for selected Pedestrian, Cyclist, and Car examples.
The maps are diagnostics of learned attention pathways and should not be
interpreted as causal explanations.
