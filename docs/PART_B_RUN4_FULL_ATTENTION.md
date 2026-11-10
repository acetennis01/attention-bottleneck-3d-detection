# Run 4: unrestricted spatial attention control

Implemented 2026-11-07 for the modified Part B KITTI R50/FPN recipe. No new
training run is launched or added to the existing queue by this implementation.

## What comparison this implements

The review alternates between “full cross-attention” and “full joint attention”.
This implementation uses **unrestricted joint-token attention with separate
modality-specific parameters**. Each of four layers takes the same pre-layer
camera and LiDAR spatial tokens, concatenates them, and computes:

```
context = concat(image_tokens, lidar_tokens)    # 256 + 256 spatial tokens
image_next = image_branch(Q=image_tokens, K=context, V=context)
lidar_next = lidar_branch(Q=lidar_tokens, K=context, V=context)
```

Each branch retains pre-norm attention, residuals, and its feed-forward network.
The final LiDAR readout queries all 512 normalized final spatial tokens instead
of eight bottleneck tokens. Its result goes through the same normalization,
BEV upsampling, zero-initialized projection and residual addition. It is not
cross-only attention, a single shared-weight joint transformer, or attention
over the full-resolution BEV map. These distinctions must appear in the paper.

There are **no bottleneck parameters or bottleneck outputs**. Attention captures
are named `image_joint`, `lidar_joint`, and `lidar_readout`; no old bottleneck
heatmap renderer should be used to interpret these maps.

## Controls kept fixed

`configs/kitti_full_attention.py` inherits the gated B8 recipe and changes only
the fusion graph, required registration import and output paths. It keeps:

- R50/FPN camera backbone, calibrated height-anchor sampler, E256, 16x16 token
  grids, four layers, eight heads and modality-specific branch/readout widths.
- The same pretrained LiDAR epoch-28 initializer; frozen LiDAR/head parameters
  and normalization buffers for all 40 epochs; no LiDAR spatial masking.
- Independent pre-fusion camera auxiliary supervision, weight .25.
- Whole-camera feature dropout .20 and explicit missing-camera residual bypass.
- Seed 0, physical batch 6, accumulation 8, optimizer/schedule, validation split,
  and clean moderate 3D AP40 checkpoint selection.

Common parameter tensors initialize identically to the gated B8 arm under the
same seed. Construction consumes the B8 reference RNG sequence before removing
latent-only tensors. This is initialization bookkeeping, not active bottlenecks.
Branch and readout parameter shapes match; the full-attention model has fewer
parameters because the bottleneck embeddings, bottleneck normalization and
gates are removed. Report actual counts rather than saying parameter-matched.
Spatial attention matrices have more entries; real FLOPs, latency and memory
still require measurement. No efficiency conclusion follows from code alone.

This control compares the entire unrestricted graph against gated bottlenecks.
The average/dropout-only/gate-only arms remain necessary to isolate gating and
dropout. The completed two-epoch KITTI pilots are not 40-epoch controls: compare
equal training budgets before making a final accuracy claim.

## Verification and next execution step

Numerical tests cover direct first-layer camera-to-LiDAR gradients, common
initialization, shared pre-layer context, full attention matrix dimensions,
camera-independent auxiliary gradients, strict checkpoint reload, zero-adapter
identity, mixed-batch fallback, and dropout. Config regression compares the
resolved recipe to the gated B8 arm.

Final CPU verification on the training host: **108 tests passed** across the new
control, gated KITTI, original V2, nuScenes, multi-scale and TransFusion checks.
Fusion-module parameter counts are 7,045,508 for full attention and 7,050,120
for gated B8 (difference 4,612). These are fusion-module counts, not total or
trainable detector counts. Evidence is saved locally in
`work_dirs/run4_implementation_20261107/`.

The new source has a separate verification directory on the GPU host:
`/home/yab/mmdetection3d/work_dirs/abfuse_run4_impl_20261107_v1`.
CPU checks use `CUDA_VISIBLE_DEVICES=''`. The active TransFusion process uses
the earlier, separate source snapshot and is not modified or interrupted.

When GPU capacity is available, run the bounded real-data preflight from the
new source root with its own output directory:

```bash
export PYTHONPATH="$PWD:/home/yab/mmdetection3d"
/home/yab/miniconda3/envs/openmmlab/bin/python \
  projects/myfusion/tools/preflight_gated_kitti.py \
  --architecture full_attention \
  --work-dir work_dirs/part_b_run4_full_attention_preflight_NEW
```

This checks four accumulated optimizer updates, transferred/frozen LiDAR state,
strict reload, detection-only gradients through both attention branches and
the readout, and real-input missing-mode feature identity. It is not benchmark
AP and cannot initialize a production training run. GPU memory feasibility for
physical batch six remains unverified until this preflight runs. If it cannot
fit, do not silently reduce only Run 4's batch size: define a matched recipe.

After a passed preflight, use the full config for a fresh equal-budget run, or
explicitly prepare the same two-epoch truncated schedule as the existing pilots.
Do not load MBT weights non-strictly into this architecture. After training,
evaluate correct/black/shuffled/explicit-missing conditions at the same selected
checkpoint, then measure compute under matched conditions.

## GPU preflight and training launch — 2026-11-09

The real-KITTI preflight passed at physical batch six: 32 microbatches / four
optimizer updates, exact LiDAR transfer and frozen-state integrity, finite
gradients through camera/both attention branches/readout, strict reload, and
explicit-missing feature identity/prediction checks. All LiDAR keep masks were
one; no hard masking was enabled. Preflight weights were not used for training.
Preflight report:
`/home/yab/mmdetection3d/work_dirs/part_b_run4_preflight_20261109/report.json`.

Fresh full training started at 16:03 Pacific on 2026-11-09 under managed user
service `abfuse-run4-training-20261109.service`, with logout persistence already
enabled. Source root is the reviewed/fixed snapshot
`/home/yab/mmdetection3d/work_dirs/abfuse_review_20261107_run3_run4`.
Production output:
`/home/yab/mmdetection3d/work_dirs/part_b_run4_full_attention_20261109`.
The resolved config, separate from the smoke config, is
`run4_resolved_config.py`; progress/integrity is `run4_training_progress.json`.

Budget: 40 epochs, 1238 microbatches per epoch, physical batch six/accumulation
eight, 49520 microbatches / 6190 optimizer updates, seed zero. Same camera
dropout, auxiliary objective and permanently frozen LiDAR/head as the Part B
KITTI recipe. The launcher counts optimizer steps and checks all 126 protected
KITTI LiDAR/head tensors and frozen normalization modes at epoch boundaries.
It selects on clean overall moderate 3D AP40, not Car-only AP or corruption.

Launch inspection reached epoch one, batch 400/1238, with 50 actual optimizer
updates, finite losses, nonzero camera/attention gradients, and intact LiDAR
masks. This is an execution-health snapshot, not a validation result.

At launch, the gated KITTI comparator had only completed its two-epoch pilot.
The full-budget comparator subsequently completed the same 40 epochs. See the
[verified matched results](PART_B_KITTI_MATCHED_RESULTS_20261109.md) and
[published records](../results/part_b_20261109/README.md).
