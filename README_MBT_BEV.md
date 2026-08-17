# Symmetric MBT BEV fusion

This is the geometry-preserving camera–LiDAR implementation used by
`configs/my_fusion_mbt_bev.py`.

Key properties:

- Implements the symmetric bottleneck update from equations (8) and (9) of
  *Attention Bottlenecks for Multimodal Fusion*.
- Uses separate image and LiDAR transformer parameters.
- Adds separate learned positional and modality embeddings.
- Uses four bottleneck tokens initialized with standard deviation `0.02`.
- Preserves the full-resolution PointPillars BEV map and standard
  `Anchor3DHead` instead of detecting directly from global bottleneck tokens.
- Uses a final bottleneck-to-LiDAR readout so the last symmetric image update
  contributes to the LiDAR-only detection representation.
- Starts as an exact LiDAR-only feature baseline through a zero-initialized
  residual fusion adapter.
- Disables geometry-changing training augmentation until synchronized
  camera–LiDAR augmentation is implemented.

Run focused tests from the MMDetection3D root:

```bash
conda activate openmmlab
PYTHONPATH="$PWD" python -m pytest \
  projects/myfusion/tests/test_mbt_bev.py -q
```

Build-check the config:

```bash
PYTHONPATH="$PWD" python - <<'PY'
from mmengine.config import Config
from mmdet3d.utils import register_all_modules
from mmdet3d.registry import MODELS
cfg = Config.fromfile('projects/myfusion/configs/my_fusion_mbt_bev.py')
register_all_modules(init_default_scope=True)
model = MODELS.build(cfg.model)
print(type(model).__name__)
PY
```

## Stable single-GPU training protocol

The upstream PointPillars recipe is named `8xb6`: its LR and AdamW decay are
calibrated for 48 samples per optimizer update.  Running it with batch size 1
and no accumulation performs 48 times as many optimizer updates per sample
window.  In the original seed-0 MBT run this drove
`fusion_module.image_projection.weight` to an L2 norm near `7e-12`, effectively
disconnecting the camera branch.

On the L4 VM, use batch size 6 and accumulate eight microbatches.  The smoke
test executes two optimizer updates, verifies memory, and prints per-component
gradient norms:

```bash
PYTHONPATH="$PWD" python tools/train.py \
  projects/myfusion/configs/my_fusion_mbt_bev_eb48_smoke.py
```

Then run the matched LiDAR control and MBT experiment:

```bash
PYTHONPATH="$PWD" python tools/train.py \
  projects/myfusion/configs/pointpillars_lidar_control_eb48.py

PYTHONPATH="$PWD" python tools/train.py \
  projects/myfusion/configs/my_fusion_mbt_bev_eb48.py
```

Both use seed 0, 80 epochs, batch size 6, eight-step accumulation, the same LR
schedule, and the same effective batch size of 48.  The MBT run additionally
logs image, LiDAR, bottleneck, and output gradient norms plus the BEV residual
ratio every 100 iterations.

The configs also correct the inherited momentum migration: MMEngine expects
absolute momentum targets, so AdamW beta1 now follows `0.95 -> 0.85 -> 0.95`
instead of approaching 1.0 and destabilizing the final optimizer update.

Audit optimizer coverage and checkpoint tensors with:

```bash
PYTHONPATH="$PWD" python projects/myfusion/tools/audit_mbt_training.py \
  projects/myfusion/configs/my_fusion_mbt_bev_eb48.py CHECKPOINT [CHECKPOINT]
```

`my_fusion_mbt_bev_lr3e4.py` is retained as a lower-LR ablation.  It should not
be used as the primary corrected experiment because lowering LR alone does not
fix the optimizer-update-frequency mismatch.

The original batch-1 command is retained only for reproducing the first run:

```bash
PYTHONPATH="$PWD" python tools/train.py \
  projects/myfusion/configs/my_fusion_mbt_bev.py
```

Run the checkpointed two-epoch diagnostic first:

```bash
PYTHONPATH="$PWD" python tools/train.py \
  projects/myfusion/configs/my_fusion_mbt_bev_smoke.py
```

It saves checkpoints after each epoch and persists formatted validation output
at `work_dirs/mbt_bev_diagnostic/pred_instances_3d.pkl`. Audit box dimensions,
bottom-center height, and horizontal matching with:

```bash
PYTHONPATH="$PWD" python projects/myfusion/tools/audit_kitti_predictions.py
```

Re-evaluate the persisted file without rerunning inference:

```bash
PYTHONPATH="$PWD" python \
  projects/myfusion/tools/evaluate_persisted_kitti.py \
  work_dirs/mbt_bev_diagnostic/pred_instances_3d.pkl
```

Before reporting benchmark results, use the upstream MMDetection3D KITTI metric
logic (or a regression-tested CPU overlap backend with identical criterion
semantics) and submit test-set detections to the official KITTI server.

## Best-checkpoint validation results

> [!IMPORTANT]
> **Camera–LiDAR MBT reaches 41.1813 overall 3D AP40 moderate, compared with
> 38.6868 for LiDAR-only PointPillars: a gain of +2.4945 AP points (+6.45%
> relative).**

### Overall comparison

| Model | Input modalities | Best epoch | Easy | **Moderate** | Hard |
| :--- | :--- | ---: | ---: | ---: | ---: |
| LiDAR-only PointPillars | LiDAR | 28 | 49.5614 | 38.6868 | 36.1066 |
| **Attention-bottleneck fusion (MBT)** | **Camera + LiDAR** | **58** | **52.1218** | **41.1813** | **38.2311** |
| **MBT improvement** |  |  | **+2.5604** | **+2.4945** | **+2.1245** |

### Moderate-difficulty breakdown

| Class | LiDAR-only | MBT | MBT - LiDAR | Result |
| :--- | ---: | ---: | ---: | :---: |
| Pedestrian | 30.3730 | **32.3544** | **+1.9814** | ↑ |
| Cyclist | **34.1321** | 33.2382 | **-0.8939** | ↓ |
| Car | 51.5553 | **57.9512** | **+6.3959** | ↑ |
| **Overall** | 38.6868 | **41.1813** | **+2.4945** | **↑** |

The improvement is consistent across the overall easy, moderate, and hard
metrics.  Most of the gain comes from **Car** detection, with a smaller gain
for **Pedestrian** and a modest regression for **Cyclist**.  Because these are
single-seed results, additional seeds are required before treating the
difference as a confidence-qualified architecture improvement.

<details>
<summary><strong>Full strict 3D AP40 results by class and difficulty</strong></summary>

| Model | Class | Easy | Moderate | Hard |
| :--- | :--- | ---: | ---: | ---: |
| LiDAR-only | Pedestrian | 34.4906 | 30.3730 | 28.2636 |
| LiDAR-only | Cyclist | 53.4429 | 34.1321 | 32.1293 |
| LiDAR-only | Car | 60.7508 | 51.5553 | 47.9270 |
| MBT | Pedestrian | **36.2324** | **32.3544** | **29.0082** |
| MBT | Cyclist | **53.5009** | 33.2382 | 30.6601 |
| MBT | Car | **66.6320** | **57.9512** | **55.0250** |

</details>

> [!NOTE]
> These are independently reproduced results on the 3,769-sample KITTI
> validation split, not official KITTI test-server scores.  Checkpoints were
> selected by overall 3D AP40 moderate.  AP40 is a detection metric—not
> classification accuracy—and uses strict IoU thresholds of `0.50` for
> Pedestrian/Cyclist and `0.70` for Car.

Reevaluation artifacts are stored under:

- LiDAR-only:
  `work_dirs/pointpillars_lidar_control_eb48_stable_momentum_seed0/best_epoch28_eval/`
- MBT:
  `work_dirs/mbt_bev_eb48_stable_momentum_seed0/best_epoch58_eval/`

## Temporal KITTI Raw visualization

`tools/render_temporal_kitti.py` renders the matched LiDAR-only and MBT best
checkpoints on a synchronized KITTI Raw drive.  Its Simple-BEV-style layout
contains raw LiDAR, LiDAR-only detections, MBT detections, tracklet ground
truth, and the front camera with projected MBT boxes.  Predictions are made
independently per frame; the video does not apply tracking or smoothing.

For the compact 154-frame `2011_09_26_drive_0005_sync` sequence, run:

```bash
cd ~/mmdetection3d
conda activate openmmlab
export PYTHONPATH="$PWD"
python projects/myfusion/tools/render_temporal_kitti.py
```

The default output is
`work_dirs/temporal_visualization_drive_0005/2011_09_26_drive_0005_lidar_vs_mbt.mp4`.
Use `--max-frames 3` for a quick rendering smoke test and `--score-thr` to
change the displayed detection threshold.
