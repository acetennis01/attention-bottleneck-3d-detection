# KITTI sensor reliability and latency results

All detection scores are moderate overall **3D / BEV AP40**, averaged over Car, Pedestrian and Cyclist on all **3,769 validation frames**. Single seed (0).

## Sensor corruption and six-epoch pilots

Reference: existing gated B8/E256 checkpoint, epoch 18. Guided and unguided pilots each completed 6 epochs, 11,136 microbatches and 1,392 optimizer updates; both use final epoch 6. Sensor encoders and detector head remained frozen. Guided adds predicted camera/LiDAR quality to the bottleneck gate; unguided trains the same quality heads with gate guidance disabled. These pilots are separate from the submitted model.

| Condition | Severity | LiDAR-only | Gated reference | Unguided pilot | Guided pilot |
|---|---:|---:|---:|---:|---:|
| clean | 0.00 | 38.6872 / 52.8851 | 42.0490 / 54.6942 | 41.4355 / 54.1730 | 41.4034 / 54.2030 |
| camera_dim | 0.25 | — | 42.0185 / 54.6309 | 41.3571 / 54.5368 | 41.3647 / 54.4777 |
| camera_dim | 0.50 | — | 41.9729 / 54.3518 | 41.3424 / 54.3952 | 41.3408 / 54.4360 |
| camera_dim | 0.75 | — | 41.1701 / 53.3474 | 40.8858 / 54.0090 | 40.7649 / 53.8244 |
| camera_blur | 0.25 | — | 40.2056 / 52.3618 | 40.1428 / 52.7711 | 40.1277 / 52.7543 |
| camera_blur | 0.50 | — | 39.0980 / 52.0647 | 40.1740 / 52.5487 | 40.0318 / 52.4538 |
| camera_blur | 0.75 | — | 39.1666 / 51.5799 | 39.8012 / 52.3326 | 39.8674 / 52.4171 |
| lidar_dropout | 0.25 | 37.3338 / 51.7076 | 40.5094 / 53.0030 | 40.3766 / 53.2276 | 40.5275 / 53.2203 |
| lidar_dropout | 0.50 | 32.2769 / 47.5808 | 36.4656 / 49.3590 | 36.6092 / 49.3236 | 36.6504 / 49.4170 |
| lidar_dropout | 0.75 | 23.9342 / 38.9036 | 27.9700 / 41.0616 | 28.6695 / 41.7906 | 28.7725 / 41.7847 |
| lidar_noise | 0.25 | 33.1330 / 48.0310 | 37.2483 / 50.0393 | 37.5646 / 49.7762 | 37.4675 / 49.7324 |
| lidar_noise | 0.50 | 21.7487 / 36.3255 | 24.9948 / 37.9275 | 25.7187 / 38.3376 | 25.7317 / 38.2817 |
| lidar_noise | 0.75 | 12.3005 / 25.8187 | 14.6586 / 26.3274 | 15.2747 / 26.4872 | 15.1680 / 26.3907 |

Severity `s`: image intensity ×(1−s); Gaussian image blur σ=12s pixels; random removal of fraction s of LiDAR points; or xyz Gaussian noise σ=0.20s metres. LiDAR-only rows use the same LiDAR corruption. Camera-only conditions leave LiDAR unchanged; no duplicate baseline evaluations were run. The reference uses unchanged gated weights with quality guidance disabled and reproduces clean reference scores exactly.

The guided pilot lowers clean 3D AP40 by 0.6456 versus the reference. Against the matched unguided pilot, gains are small and mixed; this does not establish a useful reliability-gating improvement or camera-only inference under LiDAR absence.

[All 46 evaluations and checkpoint/report hashes](comparison.json) · [Training recipe](recipe.json) · [Guided completion](guided_completed.json) · [Unguided completion](unguided_completed.json)

## Processing cost

RTX 4090, PyTorch 2.1.0+cu121, batch 1, 10 warmup calls. Latencies are median [p10, p90] milliseconds.

| Fusion | Fusion parameters (M) | Counted fusion GFLOPs | Fusion ms | Detector ms |
|---|---:|---:|---:|---:|
| Gated B8 | 7.050 | 13.09 | 3.08 [3.07, 3.10] | 15.52 [14.97, 15.90] |
| Average B8 | 7.050 | 13.09 | 2.93 [2.92, 2.95] | 15.30 [14.78, 15.59] |
| Full attention | 7.046 | 13.70 | 3.27 [3.26, 3.28] | 15.88 [15.12, 16.28] |
| Concat | 0.594 | 11.14 | 2.09 [2.08, 2.10] | 14.63 [14.09, 15.09] |
| Add | 0.529 | 11.12 | 2.37 [2.35, 2.38] | 14.95 [14.39, 15.29] |

Gated B8 has 5.6% lower fusion latency and 2.3% lower detector latency than full attention in this measurement. Concat/add are faster.

Fusion: 100 calls on cached real features from one frame, including geometric sampling. Detector: 50 calls across 10 frames, including preprocessing/voxelization through postprocessing; excludes disk I/O and dataset transforms. All arms share selected gated sensor encoders and the detector head. Average disables gating in gated weights; concat/add fusion weights are fresh. Timings compare inference graphs, not accuracy-matched trained concat/add detectors. Prediction-dependent postprocessing can affect detector timing. GFLOPs count supported fusion operators with one FLOP per fused multiply-add; unsupported operators are listed in the JSON.

[Full timing distributions, operator coverage and checkpoint hashes](latency.json)

## Reliability-head diagnosis

The trained LiDAR quality head predicts approximately 0.988 on clean input and 0.983 under 75% point dropout (target 0.25). Labels, gradients and weight updates were checked. For each modality, 62.5% of training targets are 1; a constant prediction of 1 minimizes L1 under this mixture. Sigmoid saturation further weakens quality gradients.

| LiDAR quality head | Validation weighted MAE ↓ |
|---|---:|
| Trained pilot head | 0.1894 |
| L1 refit, learning rate 1e-4 | 0.1875 |
| L1 refit, learning rate 1e-3 | 0.1041 |
| Fresh standardized token + density head, BCE, learning rate 1e-3 | 0.0635 |

Disposable head-only refits: 1,200 steps, 64 training frames and 64 separate validation frames. The density/BCE probe changes several factors together. Detector weights are unchanged; these MAE improvements have **no new detector AP result**.

[Diagnosis and probes](diagnosis/diagnosis.json) · [Optimization controls](diagnosis/optimization_controls.json)

## Evidence

Completed preflights checked strict loading, finite gate/quality gradients, frozen tensors and exact explicit-camera-absence feature identity. The archived regression log records 34 passing tests. `recipe.json` retains its original pre-launch `planned` status; the completion records and evaluation results above establish completion. `source_manifest.json` identifies the run source snapshot by hashes; it is not a claim that the working-tree prototype code has been committed. No weights, datasets, predictions or manuscript files are included. [Bundle checksums](provenance.json).
