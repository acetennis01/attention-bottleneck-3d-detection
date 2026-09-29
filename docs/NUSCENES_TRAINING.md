# nuScenes training methodology

## Dataset and controlled setup

- Dataset: official nuScenes v1.0-trainval split
- Scenes: 700 training and 150 validation
- Samples: 28,130 training and 6,019 validation
- Inputs: six synchronized cameras, the current LiDAR scan, and 10 LiDAR
  sweeps
- Classes: car, truck, trailer, bus, construction vehicle, bicycle,
  motorcycle, pedestrian, traffic cone, and barrier
- Point-cloud range: `[-50, -50, -5, 50, 50, 3]`
- Random seed: 0
- Physical batch size: 1
- Gradient accumulation: 4 steps
- Effective batch size: 4
- Training length: 24 epochs
- Validation: every epoch beginning after epoch 1
- Checkpoint-selection metric: nuScenes Detection Score (NDS)

The LiDAR baseline and MBT experiment use the same dataset split, classes,
point-cloud range, LiDAR sweeps, seed, batch size, and validation protocol.

## LiDAR baseline

Train the PointPillars baseline first:

```bash
cd ~/mmdetection3d
conda activate openmmlab
export PYTHONPATH="$PWD"

python tools/train.py \
  projects/myfusion/configs/pointpillars_lidar_control_nuscenes.py \
  --cfg-options train_cfg.val_begin=2
```

The checkpoint hook retains the checkpoint with the highest validation NDS.
The MBT configuration resolves this best checkpoint when it is loaded, so the
baseline must finish before MBT training begins.

## MBT warm-start and stable fusion training

The MBT model is initialized from the best LiDAR baseline checkpoint rather
than training its LiDAR path from scratch. The detector explicitly remaps the
standard MMDetection3D PointPillars checkpoint prefixes:

| PointPillars checkpoint | MBT detector module |
| :--- | :--- |
| `pts_voxel_encoder` | `voxel_encoder` |
| `pts_middle_encoder` | `middle_encoder` |
| `pts_backbone` | `backbone` |
| `pts_neck` | `neck` |
| `pts_bbox_head` | `bbox_head` |

This mapping transfers the PillarFeatureNet, BEV scatter, SECOND backbone,
SECONDFPN neck, and detection-head weights. The ResNet-18 camera backbone is
initialized from ImageNet, while the camera-to-BEV alignment and
attention-bottleneck fusion parameters are newly initialized.

The transferred LiDAR modules and detection head remain frozen and in
evaluation mode for the corrected run. The ImageNet camera backbone is also
kept fixed. Camera-to-BEV projection, normalized valid-only view/height
aggregation, attention-bottleneck layers, LiDAR readout, bounded residual
adapter, and camera-only auxiliary head are trained.

The AdamW fusion learning rate is `3e-4`, reduced by a cosine schedule over 24
epochs. The residual update is bounded and begins with a scale of `0.02`.
LiDAR token and BEV dropout are disabled in the stable run. Fusion statistics,
camera aggregation health, and component gradient norms are logged every 500
iterations.

Start the MBT run only after the baseline has completed:

```bash
cd ~/mmdetection3d
conda activate openmmlab
export PYTHONPATH="$PWD"

python tools/train.py \
  projects/myfusion/configs/my_fusion_mbt_bev_nuscenes.py
```

This command trains the six-camera geometry-aligned model and selects its best
checkpoint using validation NDS. Because `load_from` starts a new optimization
run, the baseline optimizer state and epoch counter are not resumed.
