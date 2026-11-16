# Gated ABFuse results

## KITTI gated bottleneck sizes

| Tokens | Best epoch | Moderate 3D AP40 | Moderate BEV AP40 |
|---:|---:|---:|---:|
| 1 | 24 | 40.9047 | 53.4498 |
| 2 | 16 | 41.6289 | 54.3788 |
| 4 | 14 | 40.7389 | 52.6713 |
| 8 | 18 | **42.0490** | **54.6942** |
| 16 | 16 | 40.9294 | 54.1006 |

Fusion width is 256 and training camera dropout is 20% for all models.
Scores use all 3,769 clean KITTI validation frames. The 8-token row is the existing reference.
[Run records](kitti_bottleneck_sizes.json)

## November 14 snapshot

Snapshot: November 14, 2026, 09:49 PST.

| KITTI model | Status | Best epoch | 3D AP40 | BEV AP40 |
|---|---|---:|---:|---:|
| Average bottleneck, no dropout | Completed 40 epochs | 14 | 41.7982 | 53.9732 |
| Gated B8/E256 + dropout | Completed reference | 18 | 42.0490 | 54.6942 |
| Average bottleneck + dropout | Training: 6/40 epochs completed | — | — | — |

AP40: moderate difficulty, mean across Car/Pedestrian/Cyclist, all 3,769
validation frames. BEV uses the checkpoint selected by 3D AP40.

New sweep: **1 completed, 1 training, 16 pending** across KITTI and nuScenes.

[Code tests](publication_tests_20261114.log): **27 passed**.

[Full validation history](kitti_average_no_dropout/summary.json) ·
[Component compute results](compute_components.json)
