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

## KITTI gate and camera dropout

Eight tokens and width 256. Camera dropout, when enabled, is 20%.

| Update | Camera dropout | Best epoch | Moderate 3D AP40 | Moderate BEV AP40 |
|---|---|---:|---:|---:|
| Average | Off | 14 | 41.7982 | 53.9732 |
| Average | On | 30 | 41.9867 | 54.2849 |
| Gated | Off | 30 | 41.4109 | 53.3112 |
| Gated | On | 18 | 42.0490 | 54.6942 |

## KITTI fusion width

Eight tokens, gated fusion and 20% camera dropout. Both sensor backbones are fixed across these runs.

| Width | Best epoch | Moderate 3D AP40 | Moderate BEV AP40 | Fusion GFLOPs |
|---:|---:|---:|---:|---:|
| 128 | 14 | 42.4193 | 54.1193 | 6.12 |
| 256 | 18 | 42.0490 | 54.6942 | 13.09 |
| 512 | 14 | 40.8755 | 53.1350 | 29.58 |

All KITTI arms completed 40 epochs with seed 0. Selection uses clean
moderate 3D AP40 on all 3,769 validation frames. BEV is from the same
checkpoint. GFLOPs count supported fusion operations at batch size one
with one FLOP per fused multiply-add.

[Validation histories and hashes](kitti_component_and_width.json) ·
[Component compute](compute_components.json) ·
[November 14 queue snapshot](snapshot_20261114/queue_status.json)
