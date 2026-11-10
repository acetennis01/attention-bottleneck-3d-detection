# Part B Run 1 completion and four-run interpretation — 2026-11-10

This follow-up adds the completed camera-absent KITTI capacity control. The
November 9 bundle and original experimental evidence are preserved unchanged.
All scores are validation results, from a single seed. No model weights,
datasets, raw predictions or full training logs are included.

## KITTI comparison

All fusion arms completed 40 epochs / 49520 microbatches / 6190 optimizer
updates. Select the best overall moderate 3D AP40 checkpoint; report BEV at
that same checkpoint. Overall averages Car/Pedestrian/Cyclist on all 3769
validation frames. These are the new frozen-detector R50/FPN E256/B8 experiments,
separate from the historical jointly tuned KITTI models.

| Model | Selected epoch | Moderate 3D AP40 | Moderate BEV AP40 |
|---|---:|---:|---:|
| Matched LiDAR initializer | 28 | 38.69 | 52.89 |
| Run 1: camera-absent capacity control | 14 | 40.35 | 52.06 |
| Run 2: gated bottlenecks with cameras | 18 | **42.05** | **54.69** |
| Run 4: unrestricted full attention | 16 | 40.33 | 53.58 |

Exact Run 1 scores: 40.3501 / 52.0620. Run 2 minus Run 1 is
**+1.6989 / +2.6322 AP points**. Run 2 minus Run 4 remains
**+1.7218 / +1.1173 points**. Scores and differences are rounded only for display.
Run 1's selected checkpoint retains all 126 protected LiDAR/head tensors exactly.

Run 1 shares the gated architecture, initializer, data, seed, batch/accumulation,
optimizer, schedule, permanent LiDAR/head freeze and selection rule with Run 2.
It always removes camera information and keeps the fusion residual trainable.
Absent cameras remove valid auxiliary supervision; the residual receives
detection gradients on all examples, whereas Run 2 bypasses whole-camera-drop
examples. It is an extra-capacity control, not perfect causal isolation.

- [Exact comparison](kitti_comparison.json)
- [Run 1 selection, checkpoint/config hashes and integrity audit](run1_camera_absent/summary.json)
- [All 20 completed validations](run1_camera_absent/validation_history.json)
- [Original selected metric log line](run1_camera_absent/selected_validation_record.txt)
- [Completion report](run1_camera_absent/training_progress.json),
  [recovery audit](run1_camera_absent/resume_report.json)
- [Initial config](run1_camera_absent/initial_config.py),
  [effective resumed config](run1_camera_absent/resumed_config.py)
- [Initial recipe check](run1_camera_absent/matched_recipe_verification.json)

The run stopped during epoch 13 and restored the epoch-12 checkpoint at an
accumulation boundary with exact model/optimizer/scheduler checks. Final training
and validation completed at 10:54:40 Pacific on November 10. Bitwise RNG
continuity is not claimed. MMEngine re-dumped the resumed config under the
original remote filename; both initial and resumed versions are exported
explicitly. The saved launch/recovery scripts retain original host paths and
are one-off provenance, not portable reproduction entry points.

## Run 3 and the robustness boundary

The [previous bundle](../part_b_20261109/README.md) contains the completed
TransFusion-L/Swin run and four full 6019-frame camera cases at one selected
checkpoint. The LiDAR baseline is 64.43 mAP / 69.24 NDS percent.

| Camera condition | Gated/dropout mAP (%) | NDS (%) |
|---|---:|---:|
| Correct | 64.79 | 69.40 |
| Black pixels, absence not signaled | 62.44 | 67.93 |
| Shuffled bundles, absence not signaled | 63.75 | 68.83 |
| Explicit missing-camera signal | 64.43 | 69.24 |

Explicit missing mode bypassed fusion and preserved exact LiDAR BEV features
on every validation frame. This trained model's inference fallback differs
from Run 1, which learns a residual without camera information. Physical image
file absence is not supported by the evaluated loading pipeline.

Black and shuffled results improve over previous ABFuse but remain below
LiDAR-only. Completing Runs 1–4 does not establish "never worse than LiDAR-only"
under undetected corruption. It supports useful camera information in the
KITTI comparison, the observed advantage over this full-attention control,
explicit missing-camera fallback, and modest improvement in the two measured
nuScenes corruptions. See the [claim review](../../docs/PART_B_CLAIM_REVIEW_20261110.md).

The bundle [manifest](provenance.json) checksums every included report/config.
