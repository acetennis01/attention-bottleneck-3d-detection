# Sensor reliability and timing reproduction

[Results](../results/sensor_reliability_20261120/README.md) contain the completed
KITTI corruption evaluations, six-epoch pilots and latency measurements. The
submitted gated model and learned-reliability pilots are separate configurations.

## Implementation

- `fusion/mbt_bev_reliability.py`: independent camera/LiDAR quality heads use
  pre-fusion token mean and standard deviation. Their scores bias each gate
  by `strength * log(q_camera / q_lidar)`. Strength zero disables guidance.
- `fusion/mbt_detector_reliability.py`: adds L1 severity supervision with weight
  0.1. Targets come from training augmentation; inference gates read predicted
  scores, not target labels. Scores are not calibrated detection probabilities.
- `transforms/sensor_reliability.py`: matched dimming, blur, point dropout and
  xyz noise. Complete LiDAR absence and camera-only detection are unsupported.
- `fusion/simple_bev_fusion.py`: geometry-matched concat/add timing controls.
  Their weights are fresh; timing does not establish trained detection accuracy.

## Matched pilot

Use the installed MMDetection3D environment described in the root README, with
this project at `projects/myfusion`. Run from the MMDetection3D root. Datasets
and checkpoint binaries are not in GitHub. Supply the verified gated epoch-18
checkpoint (SHA-256 `c88b971c894716238659d5682ef9dc0801288d590d391a2d68d00fd2388a5ca0`)
and its `gated_resolved_config.py` in the reference directory. Use a new output
directory for another run; these commands are reproduction instructions.

```bash
python projects/myfusion/tools/run_sensor_reliability.py plan \
  --root work_dirs/reliability_reproduction \
  --reference-dir work_dirs/part_b_kitti_gated_full_20261109
python projects/myfusion/tools/run_sensor_reliability.py queue \
  --root work_dirs/reliability_reproduction
```

The queue runs both preflights, both six-epoch arms and all 46 full-validation
cases. It protects sensor/head tensors, strictly reloads checkpoints, checks
optimizer updates and resumes accumulation-aligned native checkpoints. It
stops on failure. Random-state continuity across shutdown is not guaranteed.
The originally executed [resolved configs](../results/sensor_reliability_20261120/configs/)
retain their recorded host paths and match the recipe hashes. Relocate paths
for a new machine; do not silently rewrite the archived configs.

## Latency and diagnosis

```bash
python projects/myfusion/tools/benchmark_fusion_latency.py \
  --reference-dir work_dirs/part_b_kitti_gated_full_20261109 \
  --full-attention-dir work_dirs/part_b_run4_full_attention_20261109 \
  --output work_dirs/latency_reproduction.json
python projects/myfusion/tools/diagnose_sensor_reliability.py \
  --study work_dirs/reliability_reproduction \
  --output work_dirs/reliability_diagnosis
python projects/myfusion/tools/refit_reliability_controls.py \
  --study work_dirs/reliability_reproduction \
  --diagnosis-dir work_dirs/reliability_diagnosis \
  --output work_dirs/reliability_optimization_controls.json
```

Timing additionally needs the verified full-attention epoch-16 checkpoint and
`run4_resolved_config.py`. The diagnosis fits disposable heads on 64 training
frames and assesses 64 separate validation frames. It does not update detector
weights or measure AP. New density/BCE head probes are not trained detectors.

Earlier camera-only residual-weight tools (`train_kitti_camera_quality.py`,
`prepare_kitti_camera_quality.py`, `finetune_kitti_camera_quality.py`) and their
modules remain available as a separate pilot. They scale the final fusion
residual from image content, rather than biasing bottleneck gates using two
sensor scores. Their corruption names/severities differ from the November 20
study. Do not mix their weights, scores or targets with that study.
