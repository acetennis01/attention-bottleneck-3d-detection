# Gated ABFuse results

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
