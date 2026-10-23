# nuImages-pretrained Swin-Tiny camera

Separate config: `configs/transfusion_l_mbt_swint_nuscenes.py` on
`codex/transfusion-l-mbt`. Existing ResNet and historical detectors are preserved.

## What is pretrained

The camera backbone comes from the checkpoint in the
[official OpenMMLab BEVFusion training instructions](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion#training-commands):
`swint-nuimages-pretrained.pth`. It is **not a complete camera-only 3D detector**
and has no standalone nuScenes 3D mAP/NDS to reproduce.

Official artifact URL:
`https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth`

Pinned downloaded-artifact SHA-256:
`cc8189aa34e2d743e26b1e3d0c8d6c18768d87561d404f0f82b282c2d178810f`.
This records the bytes downloaded from the official HTTPS endpoint; it is not
an independently published signature or checksum.

`NuImagesSwinTransformer` changes only checkpoint initialization, not Swin's
forward computation. The file already has native MMDetection tensor names,
without a `backbone.` prefix. `convert_weights=False` is essential; the ordinary
Microsoft-Swin conversion path is inappropriate for this file.

All **187 used tensors** load strictly and are bitwise-verified. The checkpoint
contains 189 tensors: only `norm0.weight` and `norm0.bias` are excluded because
we request stage outputs 1, 2, 3 rather than stage 0. These are unused output
normalizers, not omitted transformer layers. Missing parameters, foreign keys,
shape changes and unrecognized checkpoint hashes are fatal. No position-bias
interpolation or weight resizing is performed.

## What changes and what stays the same

- Swin-Tiny replaces ImageNet ResNet-50, using stages `[2,2,6,2]`, window 7,
  channel widths `[96,192,384,768]` and output stages 1/2/3.
- The existing image FPN input widths become `[192,384,768]`; its output stays
  256 channels. The FPN itself starts fresh, not from a BEVFusion fusion model.
- Patch embedding and first Swin stage are frozen (`frozen_stages=1`). Later
  stages, image FPN, MBT and camera-only auxiliary head are trainable.
- Activation checkpointing is enabled to reduce training memory. Mean/std and
  explicit BGR-to-RGB conversion match RGB ImageNet-style normalization. The
  official BEVFusion loader decodes RGB directly; ours converts its BGR images
  exactly once in the data preprocessor instead.
- Keep the existing 450x800 resize, padded to 480x800, rather than simultaneously
  changing to the official BEVFusion crop/depth-LSS pipeline. Selected features
  are `[B,6,256,60,100]`.
- All 306 pretrained LiDAR/head tensors, nine sweeps, five point features,
  range `[-54,-54,-5,54,54,3]`, class order and frozen-LiDAR methodology remain
  unchanged.
- Symmetric MBT remains 8 tokens, 4 layers, 8 heads, width 256, a 16x16 token
  grid, and no direct camera-to-head shortcut. No hard LiDAR masking is added.
- Auxiliary camera-center loss stays 0.25. The current 40k-update candidate
  schedule remains a hypothesis, not a proven optimum.

## Checks performed (2026-10-22)

- **146 applicable regression tests passed**, including camera and early-validation tests.
  The unrelated optional KITTI overlap-backend test remains excluded.
- A reference MMDetection Swin independently loaded from the same native state
  produces bitwise-identical CPU feature outputs on a fixed input.
- Real six-view GPU smoke passed: exact checkpoint loading, actual-pixel RGB
  normalization, finite camera features, 200 finite predictions, bitwise
  unchanged zero-residual BEV features, and numerically equivalent head outputs.
- Two disposable smoke updates gave nonzero camera, FPN and bottleneck gradients
  (second-step bottleneck norm 0.27605). All frozen LiDAR/head weights and BN
  buffers remained unchanged. Nonzero gradients establish connectivity, not
  useful camera dependence or an accuracy improvement.
- Smoke weights are discarded. The full MMEngine training/save/resume smoke
  also passed: two fresh optimizer updates, exact restoration of model/AdamW
  wrapper/schedulers at batch 8, then two further updates and separate two-frame
  format-only validations. LiDAR weights and BN buffers stayed unchanged.

## Run the checks

From the MMDetection3D root with `openmmlab` active:

```bash
export PYTHONPATH="$PWD"
bash projects/myfusion/tools/setup_nuimages_camera.sh
python -m pytest projects/myfusion/tests/test_nuimages_swin.py -q
python projects/myfusion/tools/smoke_transfusion_l_mbt.py \
  --config projects/myfusion/configs/transfusion_l_mbt_swint_nuscenes.py \
  --output work_dirs/transfusion_l_mbt_swint_smoke/report.json
bash projects/myfusion/tools/run_pretrained_camera_initialization.sh
```

Smoke/evaluation commands refuse to overwrite completed report/output locations;
use a fresh `--output` or `--work-dir` when repeating them. On the PC the full
initialization test was started in tmux `transfusion-l-swint-init-test`:

```bash
tail -f ~/mmdetection3d/work_dirs/transfusion_l_mbt_swint_initialization_launch.log
```

The test runs **all 6,019 validation frames**, with both pretrained encoders and
fresh, zero-output fusion. It saves predictions and
`work_dirs/transfusion_l_mbt_swint_initialization/camera_initialization_verification.json`.
The completed test **passed**: 64.43243 mAP / 69.23597 NDS versus the measured
LiDAR-only baseline of 64.43244 / 69.23599, within the predeclared 0.01-point
tolerance.

Matching that baseline establishes that loading/connecting the camera did not
damage the detector. It does **not** measure camera-only detection, prove this
backbone is superior to ResNet, or demonstrate a fusion gain. Those require
training followed by matched validation and camera ablations.

The training recipe now validates at batch 2,000 (500 optimizer updates), then
every 10,000 batches. Non-finite gradients stop the optimizer before a step.
`training_progress.json` records actual optimizer steps and sampled component
gradient norms **after** clipping. See [full-run instructions](TRANSFUSION_L_TRAINING.md).
