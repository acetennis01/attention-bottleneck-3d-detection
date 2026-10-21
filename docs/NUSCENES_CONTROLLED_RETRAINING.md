# nuScenes controlled continuation study

Bounded diagnostic study, 2026-10-21, on Haddr's RTX 4090. These are short
continuations from the selected V2 **epoch-5 MBT checkpoint**, not training
from scratch and not a new set of complete benchmark experiments. KITTI code,
original nuScenes configurations, and existing checkpoints are preserved.

## Questions and arms

Every comparison below uses `frozen` as its reference, rather than comparing
different pairs that also differ in optimization or initialization.

| Arm | LiDAR encoder/head | Base LR | Camera auxiliary weight | Camera-dependent FPN maps |
|---|---|---:|---:|---|
| `joint` | Trainable, including BN statistics | 0.0003 | 0.25 | Level 0 |
| `frozen` | Fixed parameters **and buffers** | 0.0003 | 0.25 | Level 0 |
| `lower_lr` | Fixed | 0.0001 | 0.25 | Level 0 |
| `lower_aux` | Fixed | 0.0003 | 0.05 | Level 0 |
| `multiscale` | Fixed | 0.0003 | 0.25 | Levels 0, 1, 2 |

`joint` versus `frozen` tests the complete LiDAR unfreezing policy. It does
**not** separate weight updates from BN-statistic changes. `lower_lr` reduces
all trainable learning rates by threefold, keeping their relative multipliers.
This is a one-factor-at-a-time screen, not a factorial interaction study.

## Shared contract

- Same full epoch-5 model initialization; fresh AdamW optimizer in every arm.
  Reference SHA-256:
  `71e9086e8b5cc77ce3b4021b3cad639b085e2b52f213606385291eaa36b1e3b2`.
- Exactly **4,000 microbatches / 1,000 actual optimizer updates per arm**.
  Physical batch one, accumulation four. Constant LR; no inherited cosine
  restart, epoch-based freeze transition, or automatic extension.
- Seed zero, same full 28,130-frame training dataset, sampler, two workers,
  six calibrated camera views, sweeps, classes, ranges and safe transforms.
  The queue verifies the complete sample-order hash and the first eight
  transformed input hashes across arms. GPU kernels are not forced to be
  bitwise deterministic; this remains a single-seed exploratory comparison.
- ResNet-50/E256/B8, four symmetric fusion layers, eight attention heads,
  independent camera supervision and **no hard LiDAR masking** throughout.
- Full 6,019-frame official nuScenes validation at the final step in each arm.
  Persist predictions, per-class AP, NDS and its component errors separately
  for each arm. Select no intermediate validation peak: compare equal-budget
  endpoints, including against the original epoch-5 27.91 mAP / 45.48 NDS.
- Save recovery checkpoints every 1,000 microbatches, retaining the last two
  plus the best-NDS model. The queue does not silently resume partial arms.

The new runs address the earlier fresh-optimizer comparison confound. The
old joint run and interrupted two-epoch frozen pilot are useful context,
but are **not** matched controls for these constant-LR step-budget experiments.
The PC reboot interrupted that pilot during its second epoch; its completed
first-epoch validation was 27.27 mAP / 45.10 NDS. It is not resumed by this study.

## Multi-scale MBT readout

The original nuScenes detector fuses only the highest-resolution FPN map.
The additional variant keeps that graph unchanged and reads its **same final
shared eight-token bottleneck state** using LiDAR queries from levels 1/2.
There is no direct camera-feature residual into either new level, and upper
levels do not feed back into the symmetric transformer in this experiment.
The readout attention and normalization parameters are shared with level 0.
Each new level has a cloned LiDAR query projection and a zero-initialized
output projection. This tests readout coverage, not a new three-stream MBT.

The initializer converter checks every original tensor exactly, clones query
projections from the original trained projection, and adds only zero output
adapters. All initial features and head logits must be bitwise equal on a real
sample. Initially, gradients into the new readout are blocked by zero output
adapters; the adapters learn first. After two smoke updates, each added scale
must independently send finite nonzero detection gradients to camera features
and shared bottleneck tokens. Smoke checkpoints are never production inputs.

## Files and running

- Configs: `configs/nuscenes_controlled_{base,joint,frozen,lower_lr,lower_aux,multiscale}.py`.
- New dataset-specific model: `fusion/mbt_nuscenes_multiscale.py`.
- Guards and real-data smoke checks: `hooks/controlled_retraining.py`.
- Sequential fail-stop queue: `tools/run_nuscenes_controlled.py`.
- Regression tests: `tests/test_nuscenes_controlled.py`.

From `/home/yab/mmdetection3d` with `openmmlab` active and `PYTHONPATH="$PWD"`:

```bash
python projects/myfusion/tools/run_nuscenes_controlled.py \
  --mode smoke --output-root work_dirs/nuscenes_controlled_smoke_NEW

python projects/myfusion/tools/run_nuscenes_controlled.py \
  --mode train --output-root work_dirs/nuscenes_controlled_20261021 \
  --smoke-root work_dirs/nuscenes_controlled_smoke_NEW
```

Use new, empty output roots. Production launch requires passing smokes for
the exact same Python source manifest. Arms run sequentially; a failed arm,
changed source, wrong initialization, unfrozen protected tensor, exposure
mismatch, or missing validation artifact stops the queue. It does not schedule
a reboot recovery or automatically select and train a larger follow-up model.

Each arm records `process.json`, `controlled_report.json`, checkpoints and
`results/pred_instances_3d/results_nusc.json`; the root `suite.json` tracks
queue status and gathers the metrics. Independent correct/black/shuffled
camera evaluations should follow before claiming useful camera reliance.
Small differences still need confirmation with longer budgets and more seeds.

## Completed results

All five arms finished on **2026-10-21 at 12:37 PDT**, with exactly 1,000
optimizer updates each and matching sample/input streams. All validation
predictions and checkpoints were retained. Every frozen arm passed exact
LiDAR/head parameter and normalization-buffer checks.

| Arm | mAP (%) | NDS (%) |
|---|---:|---:|
| Original epoch-5 reference | 27.91 | 45.48 |
| `joint` | 25.7334 | 44.6136 |
| `frozen` | 27.6609 | 45.3258 |
| `lower_lr` | 28.0150 | 45.5322 |
| `lower_aux` | 27.6725 | 45.3771 |
| `multiscale` | **28.4913** | **45.7456** |

The multi-scale improvement over the matched frozen control is **0.8304 mAP
points / 0.4197 NDS points**. Its improvement over the rounded original
reference is approximately 0.58 / 0.27 points. These are validation results
from short, single-seed continuations, not from a new full training run.
The follow-up is documented in [fresh multi-scale training](NUSCENES_MULTISCALE_FRESH.md).

## Execution record, 2026-10-21

- **115 applicable regression tests passed**; the optional KITTI CPU-overlap
  backend test was excluded because its dependency is unavailable here.
- All five real-data smokes passed: eight microbatches/two optimizer updates
  each, exact common warm-start tensors, identical sampled input streams,
  finite detections and strict checkpoint reloads. Frozen arms retained every
  LiDAR/head parameter and normalization buffer exactly.
- The multi-scale real-data smoke verified identical initial features and
  head logits, both new output adapters learning, and camera/bottleneck
  gradients from each new detection scale after the two updates.
- Smoke root on the PC: `work_dirs/nuscenes_controlled_smoke_20261021a`.
- Training queue tmux session: `nuscenes-controlled-20261021`.
- Training output root: `work_dirs/nuscenes_controlled_20261021`.
- Queue log: `work_dirs/nuscenes_controlled_20261021_queue.log`.

From the PC, inspect the first arm's live iterations with:

```bash
tail -f /home/yab/mmdetection3d/work_dirs/nuscenes_controlled_20261021/joint/train_console.log
```

The queue log shows arm transitions; each arm has its own iteration log.
The tmux queue survives SSH disconnection, **not a PC shutdown/reboot**.
