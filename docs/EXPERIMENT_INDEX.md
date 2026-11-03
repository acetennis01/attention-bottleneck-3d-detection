# Experiment and code guide

Use this with the [root README](../README.md) and [verified results](../results/README.md).
All families coexist on `main`; no experimental branch checkout is required.

## Completed models used in comparisons

| Family | Camera / fusion | LiDAR / output | Entry configs |
|---|---|---|---|
| KITTI original and corrected sweep | R18, width 128; original B4 or corrected B1/2/4/8/16 | PointPillars, 384-channel single BEV | `kitti_bottleneck_b*.py` for the controlled sweep; historical original requires its saved config and versioned implementation |
| KITTI larger | R50 + image FPN, width 256, B8/B16 | Same PointPillars / Anchor3DHead | `kitti_camera_v2.py`, `kitti_camera_v2_b16.py` |
| nuScenes original | Frozen R18, width 128, B4 | HardVFE PointPillars; only highest-resolution 256-channel level updated | `my_fusion_mbt_bev_nuscenes.py` |
| nuScenes multi-scale | R50/FPN, width 256, B8 | Same PointPillars baseline; all three 256-channel levels updated | `nuscenes_multiscale_fresh.py` |
| nuScenes strong pretrained | nuImages Swin-Tiny/FPN, width 256, B8 | Public OpenMMLab TransFusion-L, single 512-channel BEV | `transfusion_l_mbt_swint_nuscenes.py` |

All use four symmetric fusion layers and eight attention heads per layer. The
image backbone, bottleneck count, residual adapter, detector and training rules
are configuration-dependent; these are not interchangeable checkpoints.

## What the fusion files do

- `mbt_bev.py`: stable original/current fusion implementation, used by the
  original nuScenes model. It is not identical to the historical KITTI graph.
- `mbt_bev_kitti_v1.py`: preserved historical KITTI graph, registry name
  `KittiV1SymmetricMBTBEVFusion`; `legacy` or corrected `valid_only` aggregation.
- `mbt_bev_camera_v2.py`: larger KITTI fusion; independent camera supervision
  before MBT, coupled training masks, uncapped BEV residual.
- `mbt_bev_nuscenes.py`: original six-camera geometric aggregation.
- `mbt_bev_nuscenes_camera_v2.py`: six-camera extension of V2.
- `mbt_nuscenes_multiscale.py`: bottleneck-mediated readout at all LiDAR scales.
- `mbt_detector.py`: combines camera/LiDAR backbones, fusion and anchor head;
  contains calibration plumbing, checkpoint-prefix mapping and auxiliary loss.
- `mbt_detector_camera_v2.py`: adds image-neck/FPN support for KITTI V2.
- `mbt_detector_nuscenes*.py`: handles six-camera inputs and pyramid interfaces.
- `transfusion_l_mbt.py`: integrates MBT before the unchanged TransFusion head,
  including the required BEV-axis conversion.
- `nuimages_swin.py`: strict native-format Swin checkpoint initialization.

The shared bottleneck tokens remain the semantic camera-to-LiDAR pathway.
Auxiliary camera logits are training supervision, not a detector shortcut.

## Why older configs remain

Base KITTI configs (`my_fusion_mbt_bev*.py`, `pointpillars_lidar_control*.py`)
are inheritance dependencies and historical controls. Removing them would break
completed experiments. The subsequently corrected camera-focused alias must not
be mistaken for the epoch-20 original KITTI model.

`*_mini*` and `*_smoke*` configs test preparation or execution, not full-split
accuracy. `*_black_eval.py` / `*_shuffled_eval.py` are inference controls, not
new training architectures. `nuscenes_camera_v2*`, frozen-continuation configs,
`nuscenes_controlled_*`, and `*_lower_lr.py` preserve pilots/controlled studies;
do not relabel them as the completed fresh multi-scale run.

The R50 `transfusion_l_mbt_nuscenes.py` is also the Swin config's parent and
an initialization candidate. Its existence does not imply a completed R50
TransFusion-L fusion training result.

## Reproducing an existing checkpoint

Use the corresponding saved resolved config, exact checkpoint SHA, matching
fusion class, dataset range/sweeps/class order and evaluator. Prefer the strict
`evaluate_kitti_camera_case.py` / `evaluate_nuscenes_camera_case.py` tools over
permissive loading. Checkpoints are kept on the training host, not GitHub.
