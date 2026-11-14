# Gated ABFuse studies: overnight results and snapshot

Snapshot: **November 14, 2026, 09:49 PST / 17:49 UTC**. Of 18 new training
arms, one KITTI arm is complete, one is training and 16 are pending. Both
previously completed gated B8/E256 references are reused. See the
[study protocol](../../docs/GATED_STUDIES_20261113.md) for the complete matrix.
This directory is a dated record; live progress remains on the training PC.

| KITTI model | Selected epoch | Moderate 3D AP40 | Moderate BEV AP40 |
|---|---:|---:|---:|
| Average bottleneck, no dropout (new, completed) | 14 | 41.7982 | 53.9732 |
| Gated B8/E256 + dropout (reused reference) | 18 | 42.0490 | 54.6942 |

Scores use all 3,769 validation frames and average Car, Pedestrian and Cyclist.
Each model uses its best clean moderate 3D checkpoint for both metrics. The new
control completed all 40 epochs / 49,520 microbatches / 6,190 optimizer updates;
126 protected LiDAR/head tensors remained unchanged. Its 20 validation records,
selected-checkpoint SHA-256, full selected metrics and raw full-split log record
are in [kitti_average_no_dropout](kitti_average_no_dropout/summary.json).
The existing reference is recorded in
[the earlier matched comparison](../part_b_20261109/kitti/comparison.json).

The reference exceeds this control by 0.2508 3D / 0.7210 BEV AP40 points, but
both gating and dropout differ. The ongoing average/dropout arm is the matched
control for isolating the gate under dropout. These single-seed numbers do not
establish statistical significance or universal robustness.

The PC shutdown interrupted the second arm before its first full recovery save.
Its partial attempt was preserved, and that arm restarted from identical
seed-zero pretrained inputs. The completed first arm was retained. At this
snapshot the restarted arm had completed six epochs / 8,200 microbatches /
1,025 updates, with epoch seven in progress. The service was active and GPU
utilization was 89%. The snapshot records an aligned epoch-4 recovery checkpoint.

## Evidence

- `configs/`: all 18 resolved production configs; hashes checked against the queue.
- `kitti_average_no_dropout/summary.json`: completed history, selection, checkpoint
  identity and completion audit; adjacent files retain the raw selected metrics.
- `snapshot_20261114/queue_status.json`: complete matrix and stage states.
- `snapshot_20261114/*/study_progress.json`: per-arm progress at export time;
  these update more frequently than the queue's stage records.
- `snapshot_20261114/*/preflight.json`: passed execution checks, not accuracy results.
- `snapshot_20261114/restart_record.json`: interrupted progress and restart reason.
- `snapshot_20261114/source_manifest_v4.json`: isolated production-source identities.
- `snapshot_20261114/provenance.json`: export time, source file hashes, study-code
  hashes, compute-report hash, GPU and service status.
- `snapshot_20261114/abfuse-gated-studies-20261113.service`: persistent queue command.
- `snapshot_20261114/recovery_tests.log`: 27 passing tests after the recovery fix.
- `publication_tests_20261114.log`: repeat verification before this publication.
- `launch_status.json` and `verification_tests.log`: earlier launch snapshot and
  26-test verification, preserved as historical records.
- `compute_components.json`: corrected component profiling on both datasets,
  including KITTI full attention and explicit native-MHA/SDPA counting.

On the PC, test from the isolated source directory with its source and MMDetection3D
root on `PYTHONPATH`:

```bash
/home/yab/miniconda3/envs/openmmlab/bin/python -m pytest -q \
  projects/myfusion/tests/test_gated_bottlenecks.py \
  projects/myfusion/tests/test_gated_configs.py \
  projects/myfusion/tests/test_gated_study_plan.py \
  projects/myfusion/tests/test_gated_compute_counts.py
```

The five published study-code files were checked byte-for-byte against the
isolated PC source during export. Checkpoint binaries, datasets, predictions
and manuscripts are not included.

## Scope of compute figures

Counts cover camera backbone/FPN and fusion only, with one fused multiply-add
counted as one FLOP. Unsupported operators remain explicit in JSON. Sparse
LiDAR encoding, voxelization, detector heads and postprocessing are excluded;
these are not complete-detector FLOPs.

Fusion latency uses 10 warmup calls and 50 synchronized CUDA calls on the
RTX 4090. Fresh candidate modules are profiled; this is not whole-detector FPS.
`peak_allocated_mib` includes the base model, cached inputs and candidate fusion
module, rather than isolated detector memory. No end-to-end efficiency claim
follows from this component report.
