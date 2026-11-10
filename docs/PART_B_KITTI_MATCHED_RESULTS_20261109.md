# Completed matched KITTI comparison — 2026-11-09

Both models completed 40 epochs, 49520 microbatches and 6190 optimizer updates.
Each validation uses all 3769 KITTI validation frames. Overall moderate AP40
is the mean across Car/Pedestrian/Cyclist, not Car-only AP. Select the best
clean moderate 3D checkpoint and report BEV from that same selected checkpoint.

| Method | Selected epoch | Moderate 3D AP40 | Moderate BEV AP40 |
|---|---:|---:|---:|
| Matched LiDAR initializer | 28 | 38.69 | 52.89 |
| Unrestricted full attention | 16 | 40.33 | 53.58 |
| Gated bottlenecks B8 | 18 | 42.05 | 54.69 |

Exact selected fusion scores: full attention 40.3272 / 53.5769; gated B8
42.0490 / 54.6942. Gated-minus-full differences calculated before rounding:
**+1.7218 / +1.1173 points**, displayed as **+1.72 / +1.12**.

The gated run finished at 23:05:48 Pacific on 2026-11-09. Both runs used the
same R50/FPN, E256, token grids, four layers/eight heads, initial LiDAR weights,
40-epoch frozen LiDAR/head parameters and normalization buffers, dropout .20,
independent camera auxiliary supervision, physical batch six/accumulation eight,
seed zero, optimizer, schedule, input pipeline and selection rule. The launcher
compared resolved configs from the actual runs, allowing differences only for
the fusion architecture/latent/gate options, registration import and output
paths. Full attention uses unrestricted joint spatial tokens with separate
modality parameters; it is not a single shared-weight transformer or cross-only
attention. Total parameter counts differ slightly, as documented in Run 4.

The 126 protected LiDAR/head tensors in **both selected checkpoints** were
independently verified against the same initializer after completion. Full
40-epoch validation histories, actual completion/update reports, resolved
configs, selected checkpoint hashes and matched-recipe proof are saved under:
`work_dirs/part_b_kitti_matched_comparison_20261109/` locally and
`/home/yab/mmdetection3d/work_dirs/part_b_kitti_matched_comparison_20261109`
on the PC. No weights or raw predictions were downloaded for this export.
The compact comparison, configs and completion records are also published in
[`results/part_b_20261109`](../results/part_b_20261109/README.md).

This is evidence that the gated-bottleneck variant outperformed this matched
full-attention control in one seed. It is not statistical significance, a
universal superiority claim for bottlenecks, isolation of gating versus dropout,
or measured compute savings. Camera corruption results for this matched KITTI
pair and further gate/dropout/concat controls remain separate work. Historical
jointly tuned KITTI B8 and the nuScenes models are different experiments.
