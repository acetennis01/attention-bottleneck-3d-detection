# Camera-focused MBT v2 (new KITTI experiment)

For the separate six-camera nuScenes V2 variants, see
[nuScenes V2 setup and checks](docs/NUSCENES_CAMERA_V2.md). Those configs do not
replace the historical nuScenes runs or change the KITTI training recipe.

This branch increases camera/fusion capacity and addresses LiDAR shortcuts.
It does **not** replace the historical KITTI token study or the nuScenes model.
There are no new full-training accuracy results yet. Graph tests and short
execution smokes cannot establish improved AP or useful camera dependence.

## Architecture and training

| Component | V2 setting |
|---|---|
| LiDAR | Existing PointPillars + SECOND + SECONDFPN, 384-channel BEV |
| Camera | ImageNet-pretrained ResNet-50, C3/C4/C5 into FPN |
| Camera feature | 256-channel P3 (stride 8), incorporating top-down C4/C5 |
| Alignment | KITTI calibration, four height samples, valid-only normalization |
| MBT | 256-D, eight shared tokens, eight heads, four symmetric layers, FFN ratio 4 |
| Spatial tokens | 16 x 16 per modality |
| Detector readout | LiDAR queries shared bottlenecks, upsample, zero-initialized 1x1 residual to 384 channels |
| Camera auxiliary task | Class-specific BEV center prediction **before** multimodal attention, focal loss weight 0.25 |
| LiDAR masking | One sampled spatial mask shared by token content and the BEV bypass; visible cells only |
| Optimization | AdamW, initial base LR 3e-4, cosine schedule, gradient clipping 10 |
| Schedule | 40 epochs; pretrained LiDAR/head frozen for epochs 1-5, then fine-tuned |
| Batch / seed | Physical batch 6, accumulation 8, effective batch 48, seed 0 |
| Checkpoint selection | Overall strict KITTI 3D AP40 moderate, validation every two epochs |

ResNet stem/C2 remain frozen; C3-C5 are fine-tuned at an initial LR of 6e-5.
Image BatchNorm parameters and running statistics stay fixed. FPN/fusion start
at 3e-4 and LiDAR/head at 1.5e-4 once unfrozen. FPN produces three levels;
only P3 is consumed, so its two unselected output convolutions do not receive
gradients (all three lateral/top-down levels contribute to P3).

Camera supervision uses only aligned visual features, positional/modality
embeddings, and ground-truth center targets. It cannot read LiDAR features or
the shared bottlenecks. It is training supervision, **not** a separate camera
feature shortcut into the detector. Supervision can still learn spatial priors;
only validation ablations can establish useful visual evidence.

All camera-to-LiDAR semantic exchange still passes through shared bottlenecks.
Both modality branches read the same pre-layer shared state; their temporary
bottleneck updates are averaged. The auxiliary logits are not detector inputs.

### Coupled masking

`CameraV2MaskingHook` sets the drop probability to 0 in epochs 1-2, 0.0333 in
epoch 3, 0.0667 in epoch 4, and 0.10 from epoch 5 onward. The schedule is
reconstructed from the resumed epoch; no extra hidden hook state is required.

Only cells with a valid camera projection can be dropped. The same mask is
upsampled with nearest-neighbor interpolation and applied to the original
LiDAR BEV bypass **and** the features used for tokenization. Dropped token
content is zeroed again after pooling to handle non-divisible pooling bins.
Position embeddings remain. There is no independent channel dropout and no
inverted-dropout rescaling. This prevents the exact hidden cell from surviving
through the bypass; neighboring/contextual LiDAR evidence remains available.

Masking is disabled in `eval()` regardless of the configured probability.
At initialization, evaluation output is exactly the transferred LiDAR BEV
because the final adapter is zero-initialized. The residual is not capped by
the later implementation's 0.1 gate. Keep clipping/diagnostics enabled and
judge stability from validation rather than assuming an unrestricted update
must improve performance.

## Files

- `fusion/mbt_bev_camera_v2.py`: new registered V2 fusion class; reuses the
  versioned KITTI sampler/blocks, adds independent visual supervision and masks.
- `fusion/mbt_detector_camera_v2.py`: opt-in image-neck support; reuses the
  existing detector's PointPillars checkpoint mapping, losses, and prediction.
- `hooks/camera_v2_masking.py`: masking warm-up/ramp schedule.
- `configs/kitti_camera_v2.py`: full experiment, separate output directory.
- `configs/kitti_camera_v2_smoke.py`: 16 iterations, two optimizer updates,
  full mask probability, no AP evaluation; effective smoke batch is also 48.
- `configs/kitti_camera_v2_{black,shuffled}_eval.py`: inference-only corruption
  evaluations with the same trained V2 checkpoint.
- `configs/kitti_camera_v2_no_mask.py`: **separate training** control, otherwise
  the same V2 architecture and training schedule.
- `tools/smoke_camera_v2.py`: real-data checks, including exact checkpoint
  transfer, updates, frozen-state integrity, unfreezing, detection-only camera
  gradients (excluding auxiliary loss), prediction, and memory.
- `tests/test_camera_v2*.py`: graph, masking, image-neck, schedule and config tests.

Historical fusion/detector source files and existing configs are unchanged.
The shared diagnostic hook only gains image-neck and mask-probability logging.

## Run from the MMDetection3D root

Keep this project at `projects/myfusion` in the existing MMDetection3D checkout:

```bash
cd ~/mmdetection3d
conda activate openmmlab
export PYTHONPATH="$PWD"
```

Confirm `load_from` in `configs/kitti_camera_v2.py` resolves to your existing
best LiDAR-only epoch-28 checkpoint. It is inherited from
`kitti_bottleneck_base.py`; override it if your checkpoint is elsewhere.
**Do not initialize from a 128-D MBT checkpoint or resume an old run.**
This is a new fusion experiment with a transferred LiDAR detector and
ImageNet-pretrained camera backbone, not random initialization of everything.

Run checks first, using a new work directory:

```bash
python -m pytest projects/myfusion/tests/test_camera_v2.py \
  projects/myfusion/tests/test_camera_v2_configs.py -q

python projects/myfusion/tools/smoke_camera_v2.py \
  --work-dir work_dirs/kitti_camera_v2_smoke_check
```

The smoke runner accepts `--baseline-checkpoint "/path/with spaces/model.pth"`.
It reads a subset without editing official metadata. Its checkpoints and
predictions are execution artifacts, not benchmark results.

Start full training **manually** after reviewing the checks:

```bash
python tools/train.py projects/myfusion/configs/kitti_camera_v2.py
```

The new work directory is `work_dirs/kitti_camera_v2_r50_e256_b8_seed0`.
Resume this experiment only with its own config and `--resume`.

### Controlled 16-token V2 run

`configs/kitti_camera_v2_b16.py` inherits the full eight-token V2 config and
changes only the bottleneck count and output paths. It starts independently
from the same LiDAR-only checkpoint, not from the trained eight-token model.
The 40 epochs, seed, batch, masking, alignment, auxiliary loss, optimizer,
freezing schedule and 3D-AP-based checkpoint selection are unchanged.

```bash
python projects/myfusion/tools/smoke_camera_v2.py --tokens 16 \
  --work-dir work_dirs/kitti_camera_v2_b16_smoke_check
python tools/train.py projects/myfusion/configs/kitti_camera_v2_b16.py
```

Its output directory is `work_dirs/kitti_camera_v2_r50_e256_b16_seed0`.
The eight-token outputs are not modified. Compare both models with the same
evaluator backend; more tokens are not assumed to improve either AP metric.

## Evaluate useful camera contribution

After training, use the **same selected V2 checkpoint** for all three:

```bash
python tools/test.py projects/myfusion/configs/kitti_camera_v2.py \
  "/path/to/best_v2_checkpoint.pth"
python tools/test.py projects/myfusion/configs/kitti_camera_v2_black_eval.py \
  "/path/to/best_v2_checkpoint.pth"
python tools/test.py projects/myfusion/configs/kitti_camera_v2_shuffled_eval.py \
  "/path/to/best_v2_checkpoint.pth"
```

Outputs are persisted under distinct `work_dirs/kitti_camera_v2*` prefixes.
Black/shuffled inputs are corruption tests, not natural-distribution benchmarks.
The target is higher AP with correct images and a reproducible reduction when
visual evidence is disrupted, **not** just a larger corruption-induced drop.

Train `kitti_camera_v2_no_mask.py` separately to see whether masking actually
helps clean-input AP. Do not report its untrained config as an ablation result.
For a paper claim specific to bottleneck fusion, also train matched simpler
fusion controls and a LiDAR-only continuation control; the additional LiDAR
fine-tuning can contribute to gains over the original baseline. Repeat key
comparisons across seeds. Larger camera features, auxiliary supervision and
masking are multiple simultaneous changes, not an isolated token-count study.
Retest bottleneck size if making size claims for this new architecture.

## Verification record

The 71 V2 graph/config and historical/current fusion regressions passed, and
the real-KITTI GPU smoke passed with 38,079,500 total model parameters.
The checks were run
in an isolated temporary checkout on Haddr's PC. Existing repositories, data,
and training outputs were not replaced. See the committed
`docs/camera_v2_smoke_report.json` for the final batch-six GPU result.
The optional three-test KITTI compatibility-backend suite requires installing
the existing `mmdet3d_patches` modules; those modules were absent on that host
and were not installed or changed for this work.

No full training or full-validation accuracy run was started by this change.
