# Verified experimental results

All values are validation results (KITTI: 3,769 frames; nuScenes: 6,019).
KITTI overall AP40 averages Car, Pedestrian and Cyclist. nuScenes mAP/NDS in
the tables below are percentages; official devkit JSON files retain unit scale.
Use one selected checkpoint for all metrics of a model, not separate metric maxima.

## November 10 follow-up

[Completed Run 1 and combined comparison](part_b_20261110/README.md) add the
full-budget camera-absent KITTI capacity control: 40.3501 / 52.0620 moderate
3D/BEV AP40. The full-camera gated model improves by +1.6989 / +2.6322 AP points.
The [four-run claim review](../docs/PART_B_CLAIM_REVIEW_20261110.md) distinguishes
the measured accuracy gains and explicit missing-camera fallback from the
unsupported claim of never underperforming LiDAR under arbitrary corruption.
The new bundle preserves the earlier bundles and includes its own hashes.

## November 9 follow-up

[Part B results](part_b_20261109/README.md) add a completed matched 40-epoch
KITTI gated-bottleneck/full-attention comparison and four fixed-checkpoint
nuScenes camera conditions. Gated KITTI scored 42.05 / 54.69 moderate 3D/BEV
AP40 versus full attention's 40.33 / 53.58. The new nuScenes model preserved
clean performance and improved black/shuffled results over previous ABFuse;
undetected corruptions still score below LiDAR-only. These are single-seed
observations. The dated bundle has its own provenance manifest; the original
reports and original manifest below are preserved.

## Main comparisons

| Experiment | Selected checkpoint | 3D AP40 moderate | BEV AP40 moderate |
|---|---|---:|---:|
| KITTI LiDAR-only, matched new evaluation | epoch 28 | 38.6872 | 52.8851 |
| KITTI original four-token fusion | epoch 20 | 42.6696 | 54.6478 |
| KITTI larger eight-token fusion | epoch 14 | **44.3949** | **55.0548** |
| KITTI larger sixteen-token fusion | epoch 14 | 43.2628 | 54.2189 |

The original historical LiDAR evaluation was 38.6868 / 52.8874; the later matched
backend evaluation above differs slightly, not by a different checkpoint.
Use the historical baseline for the historical +3.9828 / +1.7604 gain, and the
matched newer evaluation for the larger B8 +5.7077 / +2.1697 gain.

| nuScenes experiment | Selection | mAP (%) | NDS (%) |
|---|---|---:|---:|
| PointPillars LiDAR-only | best epoch 24 | 25.61393688 | 44.50138811 |
| Original four-token fusion | best NDS, epoch 9 | 27.77973649 | 45.29463811 |
| Multi-scale PointPillars / ResNet-50, B8 | best NDS, microbatch 150,000 | 30.16645388 | 46.21459791 |
| TransFusion-L LiDAR-only, reproduced | official component checkpoint | 64.43243952 | 69.23598513 |
| TransFusion-L / nuImages Swin-Tiny fusion, B8 | best NDS, microbatch 110,000 | **64.79498295** | **69.38973585** |

The stronger fusion gain is +0.36254343 mAP / +0.15375072 NDS percentage points.
Its highest-mAP checkpoint was instead microbatch 80,000; do not combine that
mAP with the selected 110,000-checkpoint NDS. Full training completed 160,000
microbatches / 40,000 optimizer updates. Initialization verification scores are
baseline-preservation checks, not trained fusion improvements.

## Controlled KITTI bottleneck sweep

ResNet-18, width 128, four layers/eight heads, 40 epochs, seed 0, same baseline
warm-start and training recipe. The larger models above are a separate family.

| Shared tokens | Best epoch | Moderate 3D AP40 | Moderate BEV AP40 |
|---:|---:|---:|---:|
| 1 | 28 | 42.7868 | 53.8465 |
| 2 | 8 | 43.0612 | 54.8251 |
| 4 | 28 | 42.6099 | 53.9047 |
| 8 | 40 | **43.2288** | 54.0672 |
| 16 | 14 | 43.0773 | **55.5481** |

More tokens do not monotonically improve AP. The corrected four-token sweep
is not the historical original paper checkpoint (42.6696).

## Camera dependence

The original **30** correct/black/shuffled evaluations across **ten** models completed
with strict state loading, unchanged weights and the full validation splits.

- [All scores](camera_ablations/SUMMARY.md)
- [Machine-readable exact scores and drops](camera_ablations/comparison.json)
- [CSV](camera_ablations/overall_metrics.csv)
- [Checkpoint/config hashes and completed-case records](camera_ablations/suite.json)

Each model directory in `camera_ablations/` contains `correct.json`, `black.json`
and `shuffled.json`. nuScenes also includes complete per-class AP and TP error
reports in `*_devkit.json`. KITTI reports contain class/difficulty/AP40 metrics.

Black means actual black pixels before normalization. Shuffle is a seeded
derangement; nuScenes uses another sample's six-camera bundle with target LiDAR
and calibration retained. These are out-of-distribution corruption controls.
Drops support camera dependence, not statistical significance, universal
robustness, or attribution of the entire baseline gain to visual information.

## Files and provenance

- `baselines/`: matched LiDAR-only evaluation reports and initializer verification.
- `saved_configs/`: exact resolved training configurations from the completed runs.
  These are provenance snapshots, not universally interchangeable entry configs.
  In particular, the original KITTI config needs its historical/versioned fusion
  implementation; do not load it through a newer class with changed parameters.
- `historical/`: original KITTI validation histories, original nuScenes camera
  comparison and custom distance-stratified AP. Range bins are not standard
  overall nuScenes metrics, and camera benefit did not increase with distance.
- `training/`: completed strong-model update count and frozen-tensor integrity.
- [provenance.json](provenance.json): SHA-256 for all exported report/config bytes
  and their source paths in the checksummed metrics-only snapshot.

These compact files were exported by `tools/curate_research_results.py` after
verifying every source against the downloaded snapshot manifest. The original
snapshot is local at `work_dirs/research_metrics_20261025`; full logs, raw
predictions, datasets and checkpoints stay outside Git. No FLOPs, latency or
multi-seed significance measurements are invented or implied.
