# Attention-bottleneck camera–LiDAR 3D detection

This MMDetection3D project contains the final KITTI experiment used in the
paper and its full nuScenes trainval adaptation: a PointPillars detector with
calibration-aware camera-to-BEV alignment and symmetric attention-bottleneck
fusion.

The final model uses:

- PointPillars with SECOND and SECONDFPN for LiDAR BEV features.
- An ImageNet-pretrained ResNet-18 camera backbone.
- KITTI calibration to sample camera features at four heights for each BEV
  location.
- Four shared bottleneck tokens, four fusion layers, and eight attention heads.
- A bottleneck-mediated LiDAR readout and a training-only camera center loss.
- A five-epoch fusion warm-up followed by joint fine-tuning for 40 epochs.

## Repository layout

```text
configs/       Final training, baseline, smoke-test, and ablation configs
fusion/        Attention-bottleneck fusion module and MMDetection3D detector
hooks/         Staged training and fusion diagnostics
transforms/    Black-image and shuffled-image evaluation transforms
tools/         Evaluation, ablation, and attention-map utilities
tests/         Focused tests for the retained implementation
docs/          Paper figures generated from the final model
```

The configuration inheritance chain is intentionally retained because these
are the exact configurations used by the experiment:

```text
my_fusion_mbt_bev.py
└── my_fusion_mbt_bev_eb48.py
    └── my_fusion_mbt_bev_aligned_eb48.py
        └── my_fusion_mbt_bev_camera_focused_eb48.py
```

## Setup

Place this repository at `mmdetection3d/projects/myfusion`, then run commands
from the MMDetection3D root:

```bash
conda activate openmmlab
export PYTHONPATH="$PWD"
```

If the VM cannot execute the upstream Numba KITTI rotated-IoU kernel, apply
the included regression-tested MMCV backend:

```bash
python projects/myfusion/mmdet3d_patches/apply.py
python -m pytest projects/myfusion/tests/test_kitti_overlap_backend.py -q
```

## Train the retained experiments

The camera-focused model warm-starts from the best LiDAR-only checkpoint at
the path specified by `load_from`, so train the baseline first:

```bash
python tools/train.py \
  projects/myfusion/configs/pointpillars_lidar_control_eb48.py
```

Run the bounded camera-focused smoke test:

```bash
python tools/train.py \
  projects/myfusion/configs/my_fusion_mbt_bev_camera_focused_eb48_smoke.py
```

Train the final fusion model:

```bash
python tools/train.py \
  projects/myfusion/configs/my_fusion_mbt_bev_camera_focused_eb48.py
```

Both full experiments use seed 0, a physical batch size of 6, and eight-step
gradient accumulation. Checkpoints are selected using overall KITTI 3D
AP40 at moderate difficulty.

## Camera ablations

Evaluate the best fusion checkpoint with correct images first, then with the
retained black-image and shuffled-image configurations:

```bash
python tools/test.py \
  projects/myfusion/configs/my_fusion_mbt_bev_camera_focused_eb48.py \
  /path/to/best_epoch_20.pth

python tools/test.py \
  projects/myfusion/configs/my_fusion_mbt_bev_camera_focused_black_eval.py \
  /path/to/best_epoch_20.pth

python tools/test.py \
  projects/myfusion/configs/my_fusion_mbt_bev_camera_focused_shuffled_eval.py \
  /path/to/best_epoch_20.pth
```

`tools/compare_camera_ablations.py` compares the persisted prediction files.
`tools/render_attention_maps.py` exports the camera and BEV attention maps used
in the paper.

## Results

On the 3,769-sample KITTI validation split, the best epoch-20 fusion checkpoint
reached **42.6696 overall 3D AP40 moderate**, compared with **38.6868** for the
LiDAR-only baseline. Correct images outperformed black images by 0.2845 points
and shuffled images by 0.5742 points at the same metric. See
[`EVALUATION.md`](EVALUATION.md) for the full results and limitations.

These are validation-split results, not official KITTI test-server scores.

## nuScenes experiment

The full nuScenes experiment uses a matched LiDAR-only PointPillars baseline
followed by camera-focused MBT fine-tuning. The MBT run automatically
warm-starts from the baseline checkpoint with the highest validation NDS,
freezes the transferred LiDAR and detection modules for five epochs, and then
jointly fine-tunes both modalities. See
[`docs/NUSCENES_TRAINING.md`](docs/NUSCENES_TRAINING.md) for the complete
training protocol and commands.
