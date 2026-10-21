# Fresh multi-scale nuScenes MBT training

This is the recommended **next full-training experiment**, not a completed
accuracy result. KITTI and all historical nuScenes experiments are retained.

## Why these changes

The completed controlled continuation study used the same epoch-5 MBT model,
fresh AdamW optimizers, 1,000 optimizer updates per arm, identical sample
streams and full 6,019-frame validation. Its measured scores were:

| Arm | mAP (%) | NDS (%) |
|---|---:|---:|
| Original epoch-5 reference | 27.91 | 45.48 |
| Joint training | 25.7334 | 44.6136 |
| Frozen LiDAR/head | 27.6609 | 45.3258 |
| Frozen + lower LR | 28.0150 | 45.5322 |
| Frozen + auxiliary weight 0.05 | 27.6725 | 45.3771 |
| Frozen + multi-scale readout | **28.4913** | **45.7456** |

The multi-scale arm gained 0.8304 mAP points over the matched frozen arm.
Lowering auxiliary supervision offered little benefit. Lower LR helped the
single-scale model; its combination with multi-scale remains untested.
These short, single-seed continuations motivate a new experiment but do not
establish an optimal full schedule, significance, or guaranteed improvement.

## Model and initialization

- ResNet-50 + image FPN, 256-dimensional features, eight shared bottleneck
  tokens, four symmetric fusion layers and eight attention heads.
- All six calibrated cameras and four height queries; existing nuScenes
  classes, splits, LiDAR sweeps, physical ranges and transforms are unchanged.
- Level 0 retains its existing symmetric MBT graph. Levels 1/2 query the same
  final bottleneck state using their own projected LiDAR features. Attention
  and normalization are shared; there is no direct image-to-head shortcut.
- Start from the **LiDAR-only best epoch-24 checkpoint**, not any MBT epoch-5,
  continuation, smoke or multi-scale initializer checkpoint. Initialize the
  camera backbone from ImageNet and the fusion/image FPN/auxiliary layers
  afresh. Thus "fresh" means new camera/fusion optimization over pretrained
  backbones, not random initialization of the entire detector.
- Freeze HardVFE, scatter, SECOND, LiDAR FPN and the detection head for the
  **entire run**, including BN means, variances and counters. Do not introduce
  an automatic unfreeze transition. Camera stem/C2 and camera BN retain their
  existing pretrained constraints; ResNet C3-C5 and the camera FPN train.
- Independent camera-BEV auxiliary supervision keeps weight **0.25**.
  Hard spatial/token/channel LiDAR masking remains disabled.
- All three output residual adapters start at zero: initial detector features
  equal the transferred LiDAR detector's features. New adapters learn first;
  camera detection gradients through them are checked after smoke updates.

`FreshFusionTrainingHook` verifies upstream `pts_*` checkpoint transfer,
checksum and every fixed LiDAR/head tensor. It rejects an MBT initializer or
nonzero cold-start adapters and counts actual optimizer steps. On a genuine
checkpoint resume it accepts already-trained fusion/optimizer state but still
requires the LiDAR branch to match its original reference exactly.

## Proposed full-run recipe

| Setting | Main config |
|---|---|
| Config | `configs/nuscenes_multiscale_fresh.py` |
| Physical batch / accumulation / effective batch | 1 / 4 / 4 |
| Seed | 0 |
| Budget | 160,000 microbatches = **40,000 optimizer updates** |
| Dataset exposure | About 5.69 passes over 28,130 training frames |
| Base AdamW LR | 0.0003; existing component multipliers retained |
| Warm-up | 2,000 microbatches / 500 updates, from 1% base LR |
| Decay | Cosine over remaining 158,000 microbatches to 1% of each group's initial LR |
| Full validation | Every 10,000 microbatches / 2,500 updates |
| Recovery checkpoints | Every 2,000 microbatches; retain latest two + best NDS |
| Output | `work_dirs/nuscenes_multiscale_fresh_r50_e256_b8_seed0` |

This replaces inherited epoch-based scheduling and the short constant-LR
continuation budget. It is a proposed, bounded starting schedule, not a claim
that 40,000 updates are necessary or sufficient. Review early validation
before deciding on extensions. Keep the current effective batch size so the
first full experiment does not also introduce a batch-size change.

Validation predictions and metrics are stored under separate
`validation/iter_010000/`, `iter_020000/`, etc., rather than overwritten.
`configs/nuscenes_multiscale_fresh_lower_lr.py` is a separate comparison with
base LR 0.0001 and correspondingly scaled cosine minimum 0.000001. It changes
no model, data, update budget or relative LR multipliers. Do not assume it
will outperform the tested-LR multi-scale recipe or automatically launch both.

## Verify and run

From the MMDetection3D root with `openmmlab` active and `PYTHONPATH="$PWD"`:

```bash
python -m pytest projects/myfusion/tests -q \
  --ignore=projects/myfusion/tests/test_kitti_overlap_backend.py

python projects/myfusion/tools/smoke_nuscenes_multiscale_fresh.py \
  --work-dir work_dirs/nuscenes_multiscale_fresh_smoke_check
```

The smoke runs eight fresh microbatches/two updates, then restores a checkpoint
into a new runner and runs eight more/two more. It verifies exact ImageNet and
LiDAR initialization, unchanged frozen buffers/weights, all-scale camera and
bottleneck detection gradients, finite predictions, strict model reload and
exact optimizer/scheduler resume. Smoke artifacts never initialize a full run.

Only after passing checks, start the full experiment explicitly:

```bash
python tools/train.py projects/myfusion/configs/nuscenes_multiscale_fresh.py
```

Use a new work directory for a fresh start; do not overwrite a previous run.
To resume an interrupted **same-recipe** full run, use the saved optimizer and
scheduler checkpoint, not a model-only best-NDS file:

```bash
python tools/train.py projects/myfusion/configs/nuscenes_multiscale_fresh.py --resume
```

The LiDAR-only reference must remain available on resume for integrity checks.
Hardcoded reference path:
`work_dirs/pointpillars_nuscenes/best_NuScenes metric_pred_instances_3d_NuScenes_NDS_epoch_24.pth`.
Expected SHA-256:
`af4443ce5eb32249fe2454d141aade6b5db93be8881169d93c65ab669aa7dfb2`.

For later camera ablations, use the new architecture-compatible
`nuscenes_multiscale_fresh_black_eval.py` and
`nuscenes_multiscale_fresh_shuffled_eval.py`, with the **same selected full-run
checkpoint** as the correct-camera evaluation. Earlier single-scale and 128-D
analysis configurations are not compatible with these new checkpoints.

No full training is automatically launched by adding or testing these files.

## Verification on Haddr's RTX 4090, 2026-10-21

**127 applicable regression tests passed**, including a CPU simulation of all
160,000 scheduler microsteps, warm-up/decay boundaries and fixed component LR
ratios. The optional KITTI CPU-overlap backend test was excluded because its
separate dependency is unavailable in this environment.

The final real-data smoke passed eight fresh microbatches and eight resumed
microbatches (four optimizer updates total). It checked exact stored ImageNet
weights/statistics, exact transfer and preservation of all 150 LiDAR/head
tensors, identical initial features at all three head scales, nonzero camera
and bottleneck gradients from all scales, 31 finite detections on its checked
frame, strict checkpoint reload and exact optimizer/scheduler resume.
Legacy ImageNet checkpoints omit BN batch counters; these were explicitly
checked to initialize to zero, not silently ignored.

Evidence: [verification summary](nuscenes_multiscale_fresh_smoke_report.json).
Full smoke artifacts are at
`/home/yab/mmdetection3d/work_dirs/nuscenes_multiscale_fresh_smoke_20261021c`.
No fresh full training or new benchmark evaluation was launched by these checks.
