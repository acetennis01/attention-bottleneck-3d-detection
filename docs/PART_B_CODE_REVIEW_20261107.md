# Runs 3 and 4 code review — 2026-11-07

Reviewed gating, full attention, geometric view exclusion, dropout, auxiliary
losses, detector integration, strict checkpoint loading, frozen modules,
initializers, schedules, accumulation, preflights and the sequential launcher.
The active Run 3 source and its eight core files were hash-matched to the local
source. No active-run files, parameters, schedules or processes were modified.

## Confirmed bug, fixed locally

**Run 4: explicit parameter reset crashes after construction.**
The full-attention class removed its bottleneck parameters but inherited a
`reset_parameters()` method that unconditionally accessed `bottleneck_tokens`.
Reproduced as an AttributeError in a numerical regression test. The override
now preserves the exact parent initialization during construction, then resets
only the common embeddings, auxiliary head and residual adapter on later calls.
Normal construction/forward did not trigger this bug; it does not affect Run 3.
Common initial parameter equality with B8 and strict reload remain verified.

## Behavior limits, not newly discovered training defects

- Complete camera dropout with exact bypass gives zero detection data gradient
  to the fusion parameters on those examples. Camera auxiliary valid cells are
  also absent. Optimizer momentum/weight decay may still update parameters.
  Partial-view dropout does supply a learning signal from surviving views.
  Do not describe the full-drop examples as teaching a learned reliability
  detector to repair corruption or as evidence of robustness to black/shuffled
  images. This intentionally implemented bypass recipe is left unchanged.
- Fusion accepts a per-sample/per-view `camera_available` signal, but current
  detector integration does not forward such a metadata/input flag. The global
  `camera_mode='missing'` config works; actual missing image files are not
  supported by the existing loading/extraction path. This was already a
  documented prototype limitation and needs integration before deployment.
- Run 4 is unrestricted joint-token attention with modality-specific weights,
  not cross-only attention. It is matched to the modified R50/frozen-detector
  KITTI recipe. Physical-batch-six GPU feasibility and real-data full-attention
  preflight remain unverified while Run 3 occupies the GPU.

## Checks and evidence

- Latest inspected Run 3 integrity report: all 306 protected LiDAR/head tensors
  unchanged, verified at stored iteration 8999. This is a dated observation.
- Shared pre-layer branch state, gate direction, direct full-attention camera
  gradients, dropout behavior, fallback, normalization and strict reload tested.
- An explicitly dropped view containing NaNs cannot affect the output from
  surviving views; its feature values are removed before geometric sampling.
- New reset test failed before the fix and passed afterward.
- **112 tests passed**, zero failures/errors/skips, using CPU-only execution on
  the host's PyTorch 2.1.0+cu121 runtime in a separate review source tree:
  `/home/yab/mmdetection3d/work_dirs/abfuse_review_20261107_run3_run4`.
- No critical normal-training bug was identified in Run 3. This audit is not
  proof of the absence of every bug, GPU capacity or improved detection quality.

The fix and added tests were prepared independently of the active experiment.
The review did not alter the running experiment or launch Run 4 training.
