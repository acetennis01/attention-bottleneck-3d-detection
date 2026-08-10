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

Train from the MMDetection3D root:

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
