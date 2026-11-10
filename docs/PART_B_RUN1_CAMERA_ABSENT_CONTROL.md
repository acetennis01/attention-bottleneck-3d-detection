# Full-budget Run 1 camera-absent control — completed 2026-11-10

This is fresh 40-epoch training from the matched LiDAR epoch-28 initializer,
not continuation of the two-epoch pilot. It uses the gated R50/FPN E256/B8
architecture and same data, seed zero, physical batch six/accumulation eight,
optimizer, 40-epoch LR schedule, permanent LiDAR/head freeze and clean moderate
3D AP40 checkpoint selection as the completed full-camera gated KITTI run.
The resolved configs were compared, allowing only the absence/bypass flags and
output paths to differ.

Camera features are always absent (`camera_mode='missing'`). The residual stays
trainable (`bypass_missing=False`), so this tests what the additional fusion
capacity can learn without images. With bypass enabled it would simply be the
fixed LiDAR baseline, with zero detection gradient to the fusion module.
Camera-drop probability is held at .20 to preserve the full-camera arm's RNG
draw/probability setting; the cameras are absent regardless of that draw.

This is not perfect causal isolation. Camera auxiliary valid cells/targets are
absent, so the auxiliary loss is zero. The residual receives detection gradients
on every training example, while the full-camera method bypasses its residual
on whole-camera-drop examples. Report those differences when interpreting the
comparison, rather than attributing the entire score difference to pixels.

A numerical check with a nonzero residual verified exact output independence
from changed images, zero image-input gradient, and nonzero adapter/bottleneck
gradients. Runtime checks assert unavailable cameras and no valid auxiliary
camera cells on every training/validation forward. Initial training confirmed
zero aligned camera content, zero auxiliary camera loss, zero image-backbone
and FPN data gradients, and finite detection losses.

The budget is 49520 microbatches / 6190 optimizer updates. All 126 protected
LiDAR/head tensors are checked against the initializer and verified at epoch
boundaries with frozen parameters and normalization buffers.

Source root:
`/home/yab/mmdetection3d/work_dirs/abfuse_review_20261107_run3_run4`.
Managed user service: `abfuse-kitti-absent-full-20261109.service`.
Output root:
`/home/yab/mmdetection3d/work_dirs/part_b_kitti_absent_full_20261109`.
Records: `absent_resolved_config.py`, `matched_recipe_verification.json`, and
`absent_training_progress.json`.

Launch time: 23:55:52 Pacific on 2026-11-09. Training stopped during epoch 13
and resumed at 09:04 Pacific on 2026-11-10 from the completed epoch-12 checkpoint
(14856 microbatches / 1857 optimizer updates). Exact model, optimizer and
scheduler restoration was checked before training. This checkpoint lies on a
complete accumulation boundary. Bitwise dropout/point-shuffle RNG continuity
across the interruption is not claimed.

Training and final validation completed at 10:54:40 Pacific on 2026-11-10:
40 epochs, 49520 microbatches and 6190 optimizer updates. All 20 scheduled
validations, at epochs 2 through 40, completed on all 3769 validation frames.
The selected epoch-14 checkpoint scored **40.3501 / 52.0620** moderate 3D/BEV
AP40. The final epoch-40 checkpoint scored 37.9367 / 50.9786; it is not the
selected model. All 126 protected tensors in the selected checkpoint were
independently rechecked against the original LiDAR initializer after completion.

The matched full-camera gated model scored 42.0490 / 54.6942 at epoch 18:
**+1.6989 / +2.6322 AP points** over this camera-absent capacity control.
This supports useful camera information beyond what the extra fusion capacity
learned without images, subject to the supervision/gradient-routing differences
above. It is a single-seed observation, not a significance or universal
robustness result. This trained residual control is distinct from the inference
fallback that bypasses fusion and returns the original LiDAR features unchanged.

Recovery service: `abfuse-kitti-absent-resume-20261110.service`.
The initial launcher and recovery driver are preserved as dated execution
snapshots, not portable commands for starting another experiment. MMEngine
re-dumped the resumed config under the original filename in the run directory.
The export therefore preserves the original local pre-resume config as
`initial_config.py` and the effective resumed version as `resumed_config.py`.

See the [completed Run 1 evidence](../results/part_b_20261110/README.md),
[four-run claim review](PART_B_CLAIM_REVIEW_20261110.md), and
[matched KITTI comparison](../results/part_b_20261110/kitti_comparison.json).
