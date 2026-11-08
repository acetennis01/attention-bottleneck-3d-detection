# Part B gated-bottleneck prototype

Prepared 2026-11-07, before pilot results were available. This is a separate experiment,
not a replacement for any completed ABFuse checkpoint or manuscript claim.

## Hypotheses and scope

Test whether learned bottleneck exchange plus camera dropout improves measured
camera-corruption behavior without sacrificing clean detection. Gating alone
does not establish novelty or guarantee improved accuracy. Missing-modality
robustness has precedents including UniBEV and MetaBEV; position results against
these and more recent robust-fusion work before changing the paper title.

The KITTI prototype uses R50/FPN, E256/B8 (the existing larger family), **not**
the R18 historical base suggested in the review. This allows independent
pre-fusion auxiliary supervision using the shared V2 graph. All five arms
start from the same LiDAR epoch-28 initializer, keep its parameters and BN
buffers frozen throughout, disable LiDAR masking, and retain the 40-epoch,
seed-0, physical-6/accumulation-8 recipe. Historical KITTI results are context,
not matched gate controls for this changed frozen-detector recipe.

## Implemented KITTI arms

| Config | Learned gate | Camera dropout | Camera input |
|---|---|---|---|
| `kitti_gated_b8.py` | yes | 0.20 | correct |
| `kitti_gated_average.py` | no | 0 | correct |
| `kitti_gated_dropout_only.py` | no | 0.20 | correct |
| `kitti_gated_gate_only.py` | yes | 0 | correct |
| `kitti_gated_absent_control.py` | yes | 0 | always absent |

Each layer's temporary bottlenecks are combined as
`b_next = g * b_camera + (1-g) * b_lidar`, with
`g = sigmoid(Linear(concat(b_lidar, b_camera)))` (one scalar per token).
Zero-initialized gate weights/bias start at equal averaging. Both branches
still read the same pre-layer shared state; no direct image-to-detector path.

Camera dropout removes aligned feature information and auxiliary valid cells.
It is **feature-level**, not raw-pixel black augmentation. A learned MISSING
token is intentionally omitted from the fallback arm: the complete residual
is bypassed for explicit absence, so such a token has no detection gradient
on missing samples. Missing samples still execute the graph; no compute-saving
claim is made. The separate `NuScenesGatedCameraMBTBEVFusion` implements
all-view dropout at .20 and independent per-view dropout at .15. Dropped views
are excluded from geometric aggregation using zero valid image extents; valid
remaining view/height weights renormalize. Zero pixels are not treated as an
availability signal. It retains the audited TransFusion/Swin initializers,
fixed detector, and 160k-microbatch / 40k-update training budget through
`transfusion_l_gated_swint_nuscenes.py`.

## Fallback versus a trainable control

`camera_mode='missing'` (whole batch) or a Boolean `camera_available` argument
(per sample) is an explicit availability signal. The existing detector can use
the config-level mode; plumbing real per-sample metadata remains future work.
The current detector still loads/processes images; physically absent image
files are not supported yet. For missing-camera evaluation use the usual data
pipeline with the explicit mode, documenting that oracle availability signal.

With fallback enabled, the **entire** fusion residual is bypassed. Merely setting
the shared bottlenecks to the LiDAR update would not recover the baseline.
Frozen detector + intact LiDAR input + bypass should reproduce LiDAR-only
predictions; this still needs real-data end-to-end verification. There is no
guarantee under undetected corruption, black pixels, shuffled images, or bad
calibration. Black pixels do not trigger a missing-camera signal automatically.

The absent-camera capacity control disables fallback to learn a LiDAR-conditioned
residual with no image information. Otherwise a fully frozen always-bypassed
model would have zero detection gradient and be a redundant baseline. This
control's absence of camera auxiliary supervision is intrinsic to the question;
it is not a perfect isolation of every optimization difference. Report that.

## Validation and experiment sequence

1. Run numerical tests for gating gradients, camera independence under explicit
   absence, exact fallback, mixed batches, eval determinism and V2 regressions.
2. On the GPU host, resolve configs; audit initializer and frozen parameters/BN;
   use a new output directory and verify strict reload, real-data forward/loss,
   optimizer updates, full-class head behavior, and absence of train masks.
3. Verify explicit missing-mode predictions against the matched LiDAR detector
   on real frames before claiming fallback equivalence.
4. Run a bounded pilot of the absent-camera control and new method, inspect
   training health, then commit to equal full training budgets for all arms.
   No auto-launch or historical PID/schedule is implied by this document.
5. Full 3,769-frame validation at the selected checkpoint for correct, black,
   shuffled and explicitly missing input. Keep corruption inputs paired across
   arms. Select by clean moderate 3D AP40 only; retain the same checkpoint across
   camera conditions. Report clean-to-corruption drops and baseline differences.
6. Train matched fusion controls, add seeds, and measure efficiency. The
   full-attention implementation is documented in
   [the Run 4 record](PART_B_RUN4_FULL_ATTENTION.md); it is not trained yet.

The user authorized KITTI pilot plus TransFusion/Swin training on 2026-11-07.
The new source is isolated on the PC under
`/home/yab/mmdetection3d/work_dirs/abfuse_part_b_20261107_gated_v1`.
`run_part_b_training.py` requires passed preflight reports, then runs a two-epoch
absent-camera pilot and a two-epoch gated/dropout pilot (full validation each
epoch), followed by the fresh full TransFusion/Swin run. The truncated pilots
retain the original 40-epoch LR schedule; they are not final matched-budget
comparisons. The existing dataset, checkpoints, and project checkout are shared
read-only inputs; all new experiment directories have the `part_b_20261107`
prefix. The launcher refuses existing output directories and stops on failures.

Launch verification on 2026-11-07: 98 GPU-host tests passed (53 numerical and
targeted regressions, 41 shared-family regressions, 4 configuration checks).
Both real-data preflights passed. KITTI trained four accumulated updates and
verified strict reload, detection-only camera/bottleneck/gate gradients, and
missing-camera feature identity on real input. TransFusion passed cold start,
four total updates across save/resume, exact optimizer/scheduler restoration,
and frozen LiDAR/head integrity. These are execution checks, not benchmark AP.

The `abfuse_part_b_training_20261107` tmux session started successfully. The
absent-camera pilot reached epoch 1, batch 150/1238 with finite logged loss
0.8742; gated KITTI and full TransFusion were pending in the sequential queue
at that inspection. Camera auxiliary loss and camera-backbone gradients were
zero in the absent-camera control as intended. This is a dated launch snapshot,
not live status. Local preflight evidence and source manifests are in
`work_dirs/part_b_20261107/`; current remote queue status is
`/home/yab/mmdetection3d/work_dirs/part_b_20261107_training_queue/status.json`.

Do not adopt the review's arbitrary one-point threshold as a significance test.
Do not claim “never worse” from a finite corruption suite, or rename the paper
“Fail-Safe” on the basis of an explicit bypass alone. Update the title and
method/results only after completed evidence supports a precise claim.
