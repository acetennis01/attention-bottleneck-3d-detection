# KITTI bottleneck smoke tests

The transferred four-token KITTI checkpoint was trained with an older fusion
implementation than the current nuScenes model. It cannot be evaluated faithfully
using `fusion/mbt_bev.py`: its camera readout is missing there, while later
normalization layers and a residual gate introduce new checkpoint parameters.

`fusion/mbt_bev_kitti_v1.py` preserves the fusion implementation from Git commit
`c2dad89`, with a distinct registry name, `KittiV1SymmetricMBTBEVFusion`. It does
not replace the current implementation. Its default `legacy` aggregation
reproduces the historical arithmetic; `valid_only` selects a parameter-free
normalization correction. The smoke runner requires a strict load
of the original four-token checkpoint to verify compatibility before proceeding.

## Source of the experiment settings

`tools/smoke_kitti_bottlenecks.py` reads the configuration stored in the checkpoint
under `meta.cfg`. The verified historical KITTI settings are:

- Four fusion layers, 128-dimensional embeddings, eight heads, four tokens.
- Training for 40 epochs, physical batch size six, accumulation of eight.
- LiDAR encoder and detection head frozen for the first **five** epochs, then
  trainable. They were not frozen for all 40 epochs.
- ResNet-18 with `frozen_stages=1`; later image backbone stages are trainable.
- LiDAR token drop probability 0.15 and BEV channel drop probability 0.10.
- Bottleneck-conditioned camera auxiliary readout, loss weight 0.25.
- Ungated residual projection; no direct local camera residual.
- AdamW learning rate 0.0003, with the original parameter-group multipliers.

These differ from the current camera-focused config, which was updated during
the later nuScenes experiments. A controlled token-count study must preserve a
single chosen architecture and recipe across all runs.

## Running the checks on Haddr

From `/home/yab/mmdetection3d`, create a new output directory and run:

```bash
bash projects/myfusion/tools/run_kitti_bottleneck_smokes.sh \
  work_dirs/kitti_bottleneck_smoke_valid_only_NEW_RUN valid_only
```

Use `legacy` as the second argument only to inspect the original numerical
behavior. The aggregation mode is recorded in every report. The corrected
suite has not yet been run on Haddr's GPU; previous saved reports are legacy.

The launcher runs each token count in a separate process: 1, 2, 4, 8, and 16.
Each run uses 96 training frames for 16 iterations (two accumulated optimizer
updates), followed by validation on 32 frames. It then probes one backward pass
with the LiDAR modules unfrozen, without an additional optimizer update. A final
32-frame inference/evaluation run loads the original four-token checkpoint.

The official validation metadata is never edited. Each smoke directory gets a
separate 32-frame annotation file to satisfy the KITTI evaluator's requirement
that annotation and prediction counts match.

Each `smoke_report.json` records:

- Strict compatibility with the historical four-token checkpoint.
- Exact LiDAR warm-start from the matched epoch-28 checkpoint.
- Unchanged frozen LiDAR parameters and BatchNorm statistics during warm-up.
- Finite losses, features, gradients, and validation metrics.
- Visible calibrated camera cells and nonzero camera/bottleneck gradients.
- Visible height-candidate weights summing to one on every forward pass,
  with minimum/maximum sums recorded (including reference inference).
- Actual parameter updates in the image backbone and fusion components.
- Gradients reaching the LiDAR backbone and detection head after unfreezing.
- Peak allocated GPU memory and successful checkpoint serialization.

AP from these subsets is only a test of the metric execution path. It must not
be used in a benchmark table or compared to the full-validation paper scores.

## Full token-count study after the corrected GPU smoke passes

The portable configs `configs/kitti_bottleneck_b{1,2,4,8,16}.py` inherit
`kitti_bottleneck_base.py`, which restores the original KITTI training recipe
from `c2dad89`. They do not inherit the later camera-focused nuScenes changes
and do not require a resolved config under `work_dirs`. The shared base uses
valid-only height normalization; the only model difference between variants
is the bottleneck-token count. Each uses seed 0, 40 epochs total, five frozen
LiDAR warm-up epochs, batch six with eight-step accumulation, and the same
baseline checkpoint, dataset split, augmentations and checkpoint metric.

From the MMDetection3D root, start the first run manually:

```bash
conda activate openmmlab
export PYTHONPATH="$PWD"
python tools/train.py projects/myfusion/configs/kitti_bottleneck_b1.py
```

Use the b2/b4/b8/b16 configs for the other runs. Work directories and
validation/test prediction prefixes are unique to each token count and include
`valid_only_seed0`. Start fresh from the epoch-28 LiDAR baseline for every run;
do not resume the original MBT checkpoint or reuse a different token count's
optimizer state. The original checkpoints and outputs are left untouched.

Train the new b4 control as well: the historical 42.6696 score belongs to the
old numerical path and cannot stand in for the corrected four-token run.
After training, compare full-validation best checkpoints under the same
metric, rerun correct/black/shuffled-camera evaluations, and profile parameters,
FLOPs and runtime consistently. No full run or profiling was launched by this
local correction.

## Local numerical regressions

With PyTorch and pytest installed (MMDetection3D is not needed for this file):

```bash
python -m pytest projects/myfusion/tests/test_kitti_v1_aggregation.py -q
```

These CPU checks cover normalization and gradients, not end-to-end GPU training
or full-dataset accuracy. The audit and preserved design decisions are in
[`KITTI_ARCHITECTURE_AUDIT.md`](KITTI_ARCHITECTURE_AUDIT.md).
