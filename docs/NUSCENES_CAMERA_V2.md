# Larger camera-focused MBT V2 on nuScenes

Branch: `codex/camera-focused-mbt-v2`. These are **new experiments**, not
replacements for the historical KITTI or nuScenes implementations. The existing
full nuScenes dataset and LiDAR baseline checkpoint are reused; no conversion
or download is required on Haddr's PC.

## Models and methodology

| Setting | Both variants |
|---|---|
| Dataset | nuScenes v1.0-trainval, 28,130 training / 6,019 validation samples |
| Inputs | Six cameras at 800 x 450 before padding; existing LiDAR sweep pipeline |
| BEV range | `[-50, -50, -5, 50, 50, 3]` meters |
| Classes | Existing nuScenes 10-class order and detection head |
| Camera | ImageNet ResNet-50 C3/C4/C5 + 256-channel FPN P3 |
| MBT | Embedding 256, four layers, eight heads, FFN ratio 4 |
| Tokens | 16 x 16 spatial tokens per modality; **8 or 16** shared bottlenecks |
| Alignment | Existing calibrated six-view/four-height sampler; valid-only softmax |
| Auxiliary task | Independent camera-only 10-class BEV center prediction, weight 0.25 |
| Residual | Zero-initialized, unrestricted projection to 256-channel LiDAR BEV |
| Optimization | AdamW LR 3e-4, cosine to 3e-6, clipping norm 10 |
| Batch | Physical 1, accumulation 4, effective 4; seed 0 |
| Schedule | 24 epochs; LiDAR/head frozen epochs 1-5, then jointly fine-tuned |
| Selection | Best validation NDS; validation starts at epoch 2, then every epoch |

ResNet stem/C2 and image BatchNorm statistics/affine parameters stay fixed.
C3-C5 fine-tune at 0.2 times the base LR, FPN/fusion at the base LR, and
LiDAR/head at 0.5 times the base LR after unfreezing. Only P3 is consumed:
all lateral/top-down FPN branches contribute, but the other two FPN output
convolutions are unused (as in KITTI V2). This is a single-GPU recipe; DDP
requires handling those unused parameters.

The nuScenes LiDAR neck returns a pyramid rather than KITTI's single
384-channel feature. Fusion updates level 0 (256 channels) and preserves the
remaining levels for the existing head. During training, **one camera-visible
spatial mask** covers the selected level's token content and residual bypass,
and is also resized onto every other direct pyramid bypass. Context from
neighboring cells remains available. This avoids leaving a separate unmasked
head-input level, but is not a claim that all LiDAR context is removed.

The mask schedule matches KITTI V2: epochs 1-2 use zero, epoch 3 uses 0.0333,
epoch 4 uses 0.0667, and epoch 5 onward uses 0.10. Evaluation disables masking.
All camera-to-LiDAR exchange still goes through shared bottleneck tokens;
auxiliary logits are supervision only, never detector inputs. There is no
direct camera residual or bounded residual gate.

## What differs from previous experiments

- Compared with **KITTI V2**: six views, nuScenes calibration, 10 classes,
  256-channel multi-level LiDAR features, nuScenes range/sweeps, effective
  batch 4, 24 epochs, and mAP/NDS instead of AP40. The V2 fusion graph is shared
  after alignment; KITTI parameter names and checkpoint shapes are unchanged.
- Compared with **historical nuScenes MBT**: ResNet-50/FPN replaces frozen
  ResNet-18; embedding grows from 128 to 256; tokens grow from 4 to 8/16;
  staged joint fine-tuning and coupled masking replace permanent freezing and
  disabled masks; the residual is no longer capped at 0.1. Do not attribute
  any eventual gain to token count alone across those older/newer recipes.
- The **8/16 V2 variants** differ only in bottleneck count and output paths.

Warm-start from the best **nuScenes LiDAR-only** checkpoint resolved by the
existing config under `work_dirs/pointpillars_nuscenes/`. Do not load KITTI
detector weights or resume a smaller nuScenes MBT checkpoint. This begins a
new fusion optimization run, not a new randomly initialized LiDAR detector.
Missing camera/fusion keys when loading LiDAR-only weights are expected; the
smoke verifies every transferred LiDAR/head tensor exactly.

## Checks before full training

From the MMDetection3D checkout containing `projects/myfusion`:

```bash
cd ~/mmdetection3d
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab
export PYTHONPATH="$PWD"

python -m pytest projects/myfusion/tests/test_nuscenes_camera_v2.py \
  projects/myfusion/tests/test_nuscenes_camera_v2_configs.py -q

python projects/myfusion/tools/smoke_nuscenes_camera_v2.py \
  --tokens 8 --work-dir work_dirs/nuscenes_v2_b8_smoke_check
python projects/myfusion/tools/smoke_nuscenes_camera_v2.py \
  --tokens 16 --work-dir work_dirs/nuscenes_v2_b16_smoke_check
```

Each smoke uses eight real samples, eight iterations and two accumulated
optimizer updates, with **no full validation**. It checks exact baseline
transfer, initial evaluation identity at every LiDAR pyramid level, frozen
LiDAR integrity, finite gradients and real camera/FPN/bottleneck/auxiliary
updates, joint detection-only backward after unfreezing, nonempty finite
predictions, disabled evaluation masks, strict checkpoint reload and memory.
Its checkpoints are execution artifacts, **not** full-training initializers.

The smoke deliberately enables the full mask immediately. On the initial
frozen detector this can cause large classification losses and pre-clipping
gradient norms. It reports no-mask warm-up and full-mask stress losses
separately. Passing means the tested computations are finite and connected;
it does **not** prove convergence, accuracy, or that the masking strength is
optimal. Keep the production ramp and monitor loss/validation when it activates.

The existing KITTI CPU-overlap-backend tests require their optional backend;
they are unrelated to nuScenes evaluation and need not alter its runtime.

### Verified execution on 2026-10-18

Both token counts passed the strengthened real-data GPU check. The applicable
regression suite passed **94 tests** (excluding the optional KITTI CPU-overlap
backend). The KITTI shared-graph refactor was also compared against the
pre-refactor code: state-dict keys and all synthetic train/eval outputs,
including attention, were bitwise identical.

| Check | 8 tokens | 16 tokens |
|---|---:|---:|
| Parameters | 39,327,859 | 39,329,907 |
| Peak allocated GPU memory, MiB | 5,731.0 | 5,731.6 |
| Initial no-mask warm-up loss | 0.890 | 0.896 |
| Immediate 0.10-mask stress loss | 433.284 | 433.290 |

The loss contrast is a warning about **abrupt masking**, not a new accuracy
result. Production does not start with that mask. A monitored pilot should
check the ramp transition before an unattended long run; finite smoke output
alone does not establish that the 0.10 mask will train well on nuScenes.
Evidence: [execution report](nuscenes_camera_v2_smoke_report.json). Full
reports/checkpoints are retained in the remote work directories listed there.

## Full training (run separately, after checks)

```bash
python tools/train.py projects/myfusion/configs/nuscenes_camera_v2.py
# After the first run completes:
python tools/train.py projects/myfusion/configs/nuscenes_camera_v2_b16.py
```

Output directories are `work_dirs/nuscenes_camera_v2_r50_e256_b8_seed0` and
`work_dirs/nuscenes_camera_v2_r50_e256_b16_seed0`. Existing experiment outputs
are not overwritten. Full training is **not** launched by adding these files.

Use the same nuScenes evaluator for baseline and fusion, then run correct,
black-camera and shuffled-camera evaluations at the selected checkpoints.
Nonzero camera gradients do not establish useful visual evidence, and extra
LiDAR fine-tuning can also contribute to gains over the original baseline.
No new nuScenes mAP or NDS results are claimed by the implementation/smokes.
