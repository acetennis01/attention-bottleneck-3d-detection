# Matched gated-ABFuse studies: 2026-11-13

This study treats the gated bottleneck model as the reference. The authorized
scope is full bottleneck-count and component sweeps on KITTI and nuScenes,
plus fusion-width and component-compute comparisons. Results and the dated
queue snapshot are in [the evidence directory](../results/gated_studies_20261113/README.md).

## Overnight result and current state

Snapshot: **2026-11-14 09:49 PST (17:49 UTC)**. One of 18 new arms is complete,
one is training, and 16 are pending. The completed KITTI average/no-dropout
control ran all 40 epochs, 49,520 microbatches and 6,190 optimizer updates.
Its selected epoch-14 checkpoint scores **41.7982 moderate overall 3D AP40**
and **53.9732 BEV AP40** on all 3,769 validation frames. Selection uses 3D AP40;
the BEV score comes from the same checkpoint. The frozen LiDAR/head audit passed.

The reused gated/dropout B8/E256 reference scores 42.0490 / 54.6942. Its
+0.2508 / +0.7210 point differences from this control combine gating and dropout;
they do not isolate gating. The average/dropout and gate/no-dropout arms complete
that comparison. These are single-seed studies.

At the snapshot, the restarted KITTI average/dropout arm had completed six
epochs and 8,200 microbatches (1,025 updates), with epoch seven in progress.
No completed nuScenes ablation or bottleneck-count result is claimed here.

## Matrix and fixed recipe

Reuse the completed gated/dropout B8/E256 checkpoint on each dataset. Train
nine additional arms per dataset, KITTI first and then nuScenes:

| Study | Additional arms | Fixed fusion settings |
|---|---|---|
| Gate × dropout | average/no dropout; average/dropout; gate/no dropout | B8, E256 |
| Bottleneck count | B1, B2, B4, B16 | gated + dropout, E256 |
| Fusion capacity | E128, E512 | gated + dropout, B8 |

The width study changes the fusion embedding width while holding both sensor
backbones fixed. It is not a larger camera-backbone or multi-scale study.
Physical batch size, optimizer, scheduler, auxiliary loss, data pipelines, full
validation split, seed zero, pretrained inputs and selection metric remain fixed
within each dataset. No preflight weights initialize production training.

| Setting | KITTI | nuScenes |
|---|---|---|
| Sensor backbones | ResNet-50/FPN + PointPillars | nuImages Swin-Tiny + TransFusion-L |
| Budget | 40 epochs / 49,520 microbatches | 160,000 microbatches |
| Physical batch / accumulation | 6 / 8 | 1 / 4 |
| Optimizer updates | 6,190 | 40,000 |
| LiDAR/head parameters and normalization buffers | frozen throughout | frozen throughout |
| Whole-camera / per-view dropout | .20 / not applicable | .20 / .15 |
| Selection on full clean validation | moderate overall 3D AP40 | NDS |

Both dropout probabilities are zero in each nuScenes no-dropout arm. The
average/dropout arm retains both. All arms retain the existing auxiliary-loss
weight .25 and disable LiDAR masking. KITTI overall AP40 averages Car,
Pedestrian and Cyclist. nuScenes metrics must be converted from fractions to
percentages when used in paper tables.

The plan derives resolved configs from the actual completed reference configs:

- `work_dirs/part_b_kitti_gated_full_20261109/gated_resolved_config.py`
- `work_dirs/part_b_20261107_transfusion_gated_full/transfusion_gated_full.py`

Every config difference is checked against intended fusion fields and operational
output/checkpoint paths. The nuScenes saved config was a resume artifact; the
plan checks its equivalence to the canonical recipe before clearing fusion
resume state. Audited initialization hooks then load official LiDAR and nuImages
weights. KITTI retains its verified LiDAR-only initializer. B8/E256 is not rerun.

## Execution, audits and shutdown recovery

PC source: `/home/yab/mmdetection3d/work_dirs/abfuse_gated_studies_20261113_source`.
Final queue: `/home/yab/mmdetection3d/work_dirs/gated_studies_20261113_v3`.
Service: `abfuse-gated-studies-20261113.service`, enabled with user lingering.
The exported unit records the exact interpreter, source path and command.

`tools/run_gated_studies.py` exposes `plan`, `preflight`, `train` and `queue`.
Each arm has separate configs, production/preflight directories, logs and audits.
A preflight verifies finite detection-only gradients into camera/bottleneck and
active gates, strict model reload, saved optimizer/scheduler state, and exact
LiDAR-feature identity in explicit camera-absence mode. These checks are not AP
results. Production audits count real optimizer updates and protect frozen tensors.

The queue skips completed arms, preserves failed attempts and stops at errors.
KITTI's 1,238 microbatches per epoch do not divide evenly by accumulation eight.
Full recovery checkpoints are therefore saved every four epochs; validation and
best-score selection remain every two. After each aligned save, the worker
archives a full checkpoint before validation can refresh MMEngine's regular
checkpoint pointer. Resume chooses the latest aligned full checkpoint and
excludes weights-only best-score files. nuScenes' 2,000-microbatch regular
interval already aligns with accumulation four. Random-state continuity across
shutdown is not claimed.

The PC shut down during the first average/dropout attempt, after two epochs and
2,500 audited microbatches, before a full recovery checkpoint existed. Its
outputs are preserved under
`interrupted_attempts/kitti_average_dropout_before_first_checkpoint_20261114`.
Only this unfinished arm was restarted from identical seed-zero pretrained
inputs on November 14; the completed control was retained. The exported restart
record documents this explicitly. The running attempt has since produced an
aligned epoch-4 checkpoint. Earlier setup attempts in v1/v2 were retained and
are excluded from final accuracy results. Source manifest v4 records the
recovery fix; optimization and selection rules did not change.

The 27-test suite covers gating, config contracts, cold initialization,
checkpoint selection and analytic attention operation counts. See the exported
logs for results and the evidence README for the exact command.

Camera corruption evaluation on newly selected checkpoints is a later evaluation
step. These architecture studies alone do not establish corruption robustness.

## Compute protocol

`tools/measure_gated_study_compute.py` profiles the camera encoder/FPN and fusion
module at actual batch-one validation feature shapes. Explicit handlers count
both native self-attention and scaled-dot-product attention, including
projections and attention matrix products. One fused multiply-add counts as one
FLOP. The JSON lists all unsupported operators and includes auxiliary logits
that the current inference graph computes.

Sparse LiDAR encoding, voxelization, detector heads and postprocessing are
excluded. Label these values **component GFLOPs**, not complete-detector FLOPs.
Fusion-only latency uses 10 warmup calls and 50 synchronized CUDA calls on an
RTX 4090. New candidate modules are freshly constructed: these timings are not
selected-checkpoint whole-detector FPS. `peak_allocated_mib` includes the base
model, cached inputs and candidate module; it is not isolated detector memory.
