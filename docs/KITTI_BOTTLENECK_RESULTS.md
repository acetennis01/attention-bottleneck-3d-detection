# KITTI bottleneck-size study

Updated 2026-11-02. These are KITTI validation-split results, not test-server
scores. Runs use the corrected `valid_only` height normalization and historical
KITTI architecture at code revision `1a4824a` on Haddr's RTX 4090.

All variants start independently from the same epoch-28 LiDAR baseline, with
seed 0, 40 epochs, batch 6, eight-step accumulation, and five frozen LiDAR
warm-up epochs. Selection uses overall moderate 3D AP40, averaged across the
three classes. BEV scores below are from that same selected checkpoint.

| Bottleneck tokens | Status | Best epoch | 3D AP40 moderate | BEV AP40 moderate |
|---|---|---:|---:|---:|
| 1 | Completed; best checkpoint re-evaluated | 28 | 42.7868 | 53.8465 |
| 2 | Completed; best checkpoint re-evaluated | 8 | 43.0612 | 54.8251 |
| 4 | Completed; best checkpoint re-evaluated | 28 | 42.6099 | 53.9047 |
| 8 | Completed; best checkpoint re-evaluated | 40 | **43.2288** | 54.0672 |
| 16 | Completed; best checkpoint re-evaluated | 14 | 43.0773 | **55.5481** |

All five runs and their correct/black/shuffled camera evaluations completed.
Exact scores, hashes and per-class reports are in [published results](../results/README.md).
The sweep's B4 run is separate from the original paper epoch-20 checkpoint.
The larger ResNet-50/FPN B8/B16 models are also a separate model family.

## One-token result

- All 40 epochs completed in approximately 2 hours 31 minutes.
- Best epoch 28: overall 3D AP40 easy/moderate/hard =
  **54.4270 / 42.7868 / 40.0709**.
- At that checkpoint, strict moderate 3D AP40: Pedestrian **31.0795**,
  Cyclist **37.4716**, Car **59.8093**.
- Epoch 40 overall moderate 3D AP40 = **42.2374**; do not substitute the final
  weights or final-epoch predictions for the selected best checkpoint.
- Separate best-checkpoint evaluation over all 3,769 validation samples
  reproduced **42.7868** exactly and persisted its predictions.
- Logged camera weight sums remained one; no crash or non-finite loss/gradient
  failure was detected. LiDAR unfreezing occurred after the five-epoch warm-up.
- Compared with the recorded LiDAR reference (**38.6868**), the improvement is
  **4.1000 AP points**. This is not isolated proof of camera benefit: the fusion
  recipe also continues LiDAR training. The completed camera ablations are
  included in the [30-case suite](../results/camera_ablations/SUMMARY.md).

## Preserved artifacts

On Haddr's PC, B1's complete original work directory remains at
`/home/yab/mmdetection3d/work_dirs/kitti_bottleneck_b1_valid_only_seed0/`.

The selected weights are
`best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_epoch_28.pth`.
`epoch_40.pth` retains final optimizer/scheduler state for recovery. Training
logs, scalar histories, the resolved config, and existing predictions are
preserved. Best-epoch predictions are separately stored under
`best_epoch_28_eval/predictions/pred_instances_3d.pkl`.

A local backup is at
`work_dirs/kitti_bottleneck_study_artifacts/b1_20261016_epoch28/` in this
repository. It includes the best and final checkpoints, logs/config/history,
and best-epoch evaluation artifacts. Checkpoint SHA-256 hashes were verified
against Haddr's files. Redundant epoch-38/39 checkpoints remain on Haddr but
were not duplicated locally. The backup is ignored by Git; weights and raw
predictions should not be added to GitHub.

B2 writes separately to
`/home/yab/mmdetection3d/work_dirs/kitti_bottleneck_b2_valid_only_seed0/`.
It does not resume B1 or overwrite its artifacts.
