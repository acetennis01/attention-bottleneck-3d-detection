# Pretrained TransFusion-L + attention bottlenecks

Branch: `codex/transfusion-l-mbt`. This is a separate experiment; no historical
KITTI, PointPillars, or multi-scale MBT implementation is replaced.

## Starting point

Use the **OpenMMLab MMDetection3D BEVFusion LiDAR-only checkpoint**, not our
PointPillars checkpoint and not a camera-fused BEVFusion checkpoint.

- [Model zoo and operator instructions](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion)
- [Original BEVFusion repository](https://github.com/mit-han-lab/bevfusion)
- OpenMMLab reference: **64.9 mAP / 69.6 NDS**, nuScenes validation.
- MIT's separate TransFusion-L release reports **64.68 / 69.28**. Do not mix
  that checkpoint's scores with this reproduction.
- Downloaded file: `bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth`.
- SHA-256: `2628f933277774e194932815ea1a5d66a0bd9f69ebc8d9040e8c92760bb6652e`.
- Installed upstream MMDetection3D revision:
  `fe25f7a51d36e3702f961e198894580d83c4387b` (v1.4.0).

The detector keeps the official five point features, **nine historical sweeps**
(ten scans including the current frame), 0.075 x 0.075 x 0.2 m voxels,
`[-54,-54,-5,54,54,3]` range, SparseEncoder, SECOND, SECONDFPN and complete
TransFusion detection head. The head consumes one **512-channel** BEV map.
The old PointPillars four-feature/three-level-head configuration is not reused.

## Integration

Six RGB views -> ImageNet ResNet-50 + FPN -> calibrated BEV sampling -> symmetric
MBT with 8 shared tokens, 4 layers, 8 heads and 256-dimensional embeddings.
LiDAR tokens read the final bottlenecks, and a zero-initialized projection adds
the result to the pretrained 512-channel neck map before its existing head.

All camera-to-detection exchange goes through bottlenecks. The independent
camera-center auxiliary loss retains weight 0.25. Hard LiDAR masking is off.
LiDAR encoder, decoder and detection head weights **and BN statistics remain
frozen**. The frozen detection head still passes gradients to the fusion input.

The official voxelizer uses `(x,y,z)` coordinates and a `B,C,X,Y` BEV tensor.
Our sampler expects `B,C,Y,X`: transpose inside fusion and transpose back before
the unchanged detection head. A rectangular-coordinate unit test and a real
CUDA voxelizer test check this rather than relying on square tensor shapes.

The public checkpoint stores spconv2 convolution kernels as `[out,kX,kY,kZ,in]`.
The installed legacy MMCV sparse backend expects `[kX,kY,kZ,in,out]`. The audited
loader explicitly permutes 21 sparse kernels, checks the inverse permutation,
and strictly loads/verifies all 306 pretrained tensors. This does not resize,
retrain, or randomly initialize any LiDAR/head weights. Every converted key is
recorded. Other shape mismatches and unexpected tensors are rejected.

## Setup and verification

Run from the MMDetection3D root on the PC, with `openmmlab` active:

```bash
export PYTHONPATH="$PWD"
bash projects/myfusion/tools/setup_transfusion_l.sh download
```

Compile the **original BEVFusion project operators**, not MMCV's differently
ordered voxelizer. No pip package upgrade or environment replacement is needed.
On this PC the system CUDA 13.4/GCC 16 are incompatible with torch 2.1/cu121, so
the build tools were installed into an isolated directory:

```bash
/home/yab/miniconda3/bin/conda create -y \
  --prefix "$PWD/work_dirs/transfusion_l_build_tools" \
  --override-channels -c nvidia/label/cuda-12.1.1 -c conda-forge \
  cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-cccl=12.1 gxx_linux-64=12

export CUDA_HOME="$PWD/work_dirs/transfusion_l_build_tools"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-c++"
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
bash projects/myfusion/tools/setup_transfusion_l.sh build
```

This compiles project-local `.so` files in place; it does not install a new torch,
MMCV, MMDetection or MMDetection3D. Existing matching CUDA library headers from
the torch wheel are used only during compilation. Once built, the compiler
environment variables are not required for inference/training.

```bash
python -m pytest projects/myfusion/tests/test_transfusion_l_mbt.py -q
python projects/myfusion/tools/smoke_transfusion_l_mbt.py
python projects/myfusion/tools/evaluate_transfusion_l.py
```

The GPU smoke uses real data to check identical LiDAR point inputs, bitwise
unchanged zero-residual BEV features and head parameters, numerically equivalent
head outputs, finite predictions, camera/bottleneck gradients, and frozen weights
and BN buffers after two disposable updates. Smoke-updated weights are discarded.
Head output comparisons explicitly report numerical tolerance, not bitwise identity.

The evaluator runs **all 6,019 validation frames**, saves predictions and metrics,
and creates `work_dirs/transfusion_l_pretrained_baseline/baseline_verification.json`.
It refuses to overwrite an existing evaluation. Reference tolerance is 0.5
percentage points for each metric; a mismatch requires diagnosis, not adjusting
the score. Report the **measured** baseline when comparing our fusion.

The evaluation pipeline explicitly chooses the nearest nine sweeps
(`test_mode=True`) for repeatable inputs. This differs from the upstream config's
default random sweep selection; baseline and fusion use the identical choice.
Data root defaults to `data/nuscenes_full/`; override using `NUSCENES_DATA_ROOT`.
Existing v2 annotations and raw data are reused without modification.

## Training is gated, not started by setup

```bash
python tools/train.py projects/myfusion/configs/transfusion_l_mbt_nuscenes.py
```

The candidate config requires a **passed full baseline report**, the exact public
checkpoint and unchanged validation annotations. Its initialization hook performs
the audited LiDAR transfer on a cold start (`load_from=None` is intentional),
but never overwrites a resumed model. The hook rejects changed pretrained tensors
and checks them every 1,000 batches and at completion. Standard `--resume` restores
native model/optimizer/scheduler state; cold starts always use public LiDAR weights,
not smoke weights or previous PointPillars MBT weights.

The initial schedule hypothesis is 160,000 batches / 40,000 optimizer updates,
batch 1 with accumulation 4, peak AdamW LR 1e-4, image-backbone LR multiplier 0.2,
500-update warmup and cosine decay. Validation every 10,000 batches saves separate
prediction folders; recovery checkpoints every 2,000 batches plus best NDS.
This budget is not claimed to be optimal. Fusion training omits object pasting and
LiDAR-only geometric augmentation to keep camera calibration valid.

## Verification status (2026-10-22)

- 137 applicable tests passed (including 10 new integration tests); the unrelated
  optional KITTI overlap-backend test was excluded because its separate backend
  is not installed. Warnings were existing matplotlib deprecations and the
  microbatch-scheduler ordering warning.
- GPU smoke passed: 200 finite predictions, identical point inputs and
  zero-residual BEV features, all pretrained head parameters identical.
- After the zero-output adapter's first update, the bottleneck gradient norm was
  0.3056; camera-backbone and image-FPN gradients were also nonzero. This checks
  connectivity, not whether camera information improves validation performance.
- All pretrained LiDAR/head tensors and BN buffers remained unchanged after the
  two discarded smoke updates. Torch peak allocated memory was 3.72 GiB; this
  does **not** include all workspace memory allocated directly by sparse CUDA ops.
- The standalone full baseline evaluation was started in tmux session
  `transfusion-l-baseline`. Its scores are **not yet a verified result**.
- On the PC, monitor with:

  ```bash
  tail -f ~/mmdetection3d/work_dirs/transfusion_l_pretrained_baseline_launch.log
  ```

The full runner/optimizer-resume training smoke remains a next milestone after
baseline reproduction; the two direct smoke updates do not replace it.

## Paused previous ablations

The previous multi-scale PointPillars ablations were stopped on 2026-10-22:

- Correct-camera full evaluation completed and reproduced **30.16645 mAP /
  46.21460 NDS**.
- Black-camera evaluation was interrupted; shuffled-camera evaluation had not run.
- All existing outputs remain in
  `work_dirs/nuscenes_multiscale_fresh_ablations_best150k_20261022/` on the PC.
- `suite.json` and `pause_record.json` record the pause. Evaluation has no
  mid-pass prediction checkpoint, so restart the interrupted condition in a fresh
  output folder later; do not rerun the original queue over existing outputs.
