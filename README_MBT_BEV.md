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
