# Part B follow-up results — 2026-11-09

These are new experiments, separate from the original 65-file export. Original
reports and `results/provenance.json` are unchanged. This directory contains
metrics/config snapshots and provenance, not weights, datasets, full training
logs, or raw predictions. See [provenance.json](provenance.json) for file hashes.

## Matched KITTI fusion comparison

Both new models completed 40 epochs / 49520 microbatches / 6190 optimizer updates.
They share the R50/FPN, initializer, frozen LiDAR/head parameters and normalization
buffers, camera dropout .20, auxiliary supervision, seed zero, physical batch
six/accumulation eight, optimizer, schedule, data and checkpoint-selection rule.
Select on clean overall moderate 3D AP40 and report BEV at the same checkpoint.
All 3769 validation frames are evaluated; overall means Car/Pedestrian/Cyclist.

| Model | Selected epoch | Moderate 3D AP40 | Moderate BEV AP40 |
|---|---:|---:|---:|
| Matched LiDAR initializer | 28 | 38.69 | 52.89 |
| Full attention | 16 | 40.33 | 53.58 |
| Gated bottlenecks B8 | 18 | **42.05** | **54.69** |

Exact new scores: full attention 40.3272 / 53.5769; gated B8 42.0490 / 54.6942.
Differences before rounding: **+1.7218 / +1.1173 AP points**. All 126 protected
LiDAR/head tensors in both selected checkpoints were rechecked against the
initializer. Full attention uses unrestricted joint spatial-token attention
with separate modality weights, not a single shared-weight transformer or
cross-only attention. Gated B8 also differs by its gate and latent/readout
mechanism; this result does not isolate gating from the bottleneck restriction.

- [Comparison, histories, selected checkpoint hashes](kitti/comparison.json)
- [Resolved-recipe equality check](kitti/matched_recipe_verification.json)
- [Full-attention config](kitti/full_attention_config.py),
  [gated config](kitti/gated_b8_config.py)
- Selected validation records preserve the exact metric log lines. Completion
  and GPU-preflight reports are also included under `kitti/`.

Historical larger KITTI B8 (44.3949) used joint LiDAR fine-tuning and a different
masking recipe; it is not the matched comparator for these frozen-detector runs.

## nuScenes TransFusion/Swin gated fusion

Training completed 160000 microbatches / 40000 optimizer updates. Selected
checkpoint: best completed clean-validation NDS, iteration 110000, SHA-256
`691beda2e84cf33c64ebf26fc513b1a20717dfd4d48443c728aed3628e195639`.
All four camera conditions use these same weights and all 6019 validation frames.
Tables use percentages; devkit reports and per-case `metrics.json` use unit scale.

| Condition | Previous ABFuse mAP / NDS (%) | Gated/dropout mAP / NDS (%) |
|---|---|---|
| Correct | 64.79 / 69.39 | 64.79 / 69.40 |
| Black pixels | 61.84 / 67.69 | **62.44 / 67.93** |
| Shuffled six-view bundle | 63.37 / 68.67 | **63.75 / 68.83** |
| Explicit camera absence | Not evaluated | 64.43 / 69.24 |

Black improvements: **+.60 mAP / +.24 NDS points**. Shuffled improvements:
**+.37 / +.16 points**. Clean performance is essentially unchanged. Differences
are calculated before rounding. Black/shuffle keep target LiDAR, calibration
and GT, and do not trigger the explicit absence signal. Missing uses an oracle
availability signal and verifies exact LiDAR BEV identity on every frame; its
scores numerically match the measured baseline within the declared tolerance.
Both undetected-corruption conditions remain below the clean LiDAR baseline.

- [Exact comparisons and drops](nuscenes/camera_ablations/comparison.json)
- [Completed evaluation manifest](nuscenes/camera_ablations/status.json)
- Per-condition reports/devkit summaries under `nuscenes/camera_ablations/`
- [Saved training config](nuscenes/selected_config.py),
  [completion/update counts](nuscenes/training_progress.json),
  [frozen integrity](nuscenes/frozen_integrity.json)
- [Validation history and interruption notes](nuscenes/validation_history.json)

The final iteration-160000 scores are 64.7992354940 mAP / 69.3607226785 NDS
percent, distinct from the selected iteration-110000 scores. Training had
interruptions; weights, optimizer and scheduler were restored on resume.
The scheduled iteration-100000 validation was interrupted and the manual resume
continued without repeating it. Selection is among completed validations.
Bitwise augmentation-RNG reproduction across interruptions is not claimed.
Recorded configs/queue scripts retain original host paths as provenance; they
are snapshots, not universally portable entry commands.

## Claim boundaries

These are single-seed observations. They support the measured matched KITTI
advantage and modest improvement on the two nuScenes corruption conditions.
They do not establish statistical significance, universal bottleneck superiority,
automatic failure detection, “never worse than LiDAR”, or end-to-end compute
savings. Gate/dropout isolation, concat controls, matched KITTI camera ablations,
multi-seed tests and FLOPs/latency/memory benchmarks remain separate work.
