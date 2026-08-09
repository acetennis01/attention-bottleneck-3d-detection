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
/home/abhiramannaluru/mbt/bin/python -m pytest \
  projects/myfusion/tests/test_mbt_bev.py -q
```

Build-check the config:

```bash
/home/abhiramannaluru/mbt/bin/python - <<'PY'
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS
cfg = Config.fromfile('projects/myfusion/configs/my_fusion_mbt_bev.py')
init_default_scope(cfg.default_scope)
model = MODELS.build(cfg.model)
print(type(model).__name__)
PY
```

Train from the MMDetection3D root:

```bash
/home/abhiramannaluru/mbt/bin/python tools/train.py \
  projects/myfusion/configs/my_fusion_mbt_bev.py
```

Before reporting benchmark results, evaluate from an unmodified MMDetection3D
KITTI evaluator and submit test-set detections to the official KITTI server.
