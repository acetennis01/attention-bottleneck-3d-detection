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
at `work_dirs/mbt_bev_diagnostic/pred_instances_3d.pkl`.

Run the matched LiDAR-only diagnostic before either full experiment:

```bash
PYTHONPATH="$PWD" python tools/train.py \
  projects/myfusion/configs/pointpillars_lidar_control_smoke.py
```

The full controlled comparison uses a shared seed (`0`), batch size (`1`),
80-epoch schedule, KITTI split/classes/range, and geometry-safe LiDAR pipeline:

```bash
# LiDAR-only control
PYTHONPATH="$PWD" python tools/train.py \
  projects/myfusion/configs/pointpillars_lidar_control.py

# Camera-LiDAR MBT
PYTHONPATH="$PWD" python tools/train.py \
  projects/myfusion/configs/my_fusion_mbt_bev.py
```

Both configs retain the last three checkpoints, save the best checkpoint by
overall moderate 3D AP40, and persist the latest formatted validation
predictions in their respective work directories.

Audit box dimensions, bottom-center height, and horizontal matching with:

```bash
PYTHONPATH="$PWD" python projects/myfusion/tools/audit_kitti_predictions.py
```

Re-evaluate the persisted file without rerunning inference:

```bash
PYTHONPATH="$PWD" python \
  projects/myfusion/tools/evaluate_persisted_kitti.py \
  work_dirs/mbt_bev_diagnostic/pred_instances_3d.pkl
```

### KITTI evaluator compatibility

The VM's Numba runtime emits PTX 8.5 while its driver accepts PTX 8.4, so the
upstream Numba rotated-IoU kernel cannot compile. Apply the included MMCV
compatibility backend from the MMDetection3D root:

```bash
python projects/myfusion/mmdet3d_patches/apply.py
PYTHONPATH="$PWD" python -m pytest \
  projects/myfusion/tests/test_kitti_overlap_backend.py -q
```

This leaves MMDetection3D's KITTI metric logic unchanged and replaces only the
incompatible overlap kernel. The MMCV backend is tested against the Shapely
reference for IoU, both intersection fractions, and raw intersection area.

### Two-epoch diagnostic result

The validated two-epoch run produced strict 3D AP40 of 49.15/41.14/38.93 for
Car, 14.18/9.21/8.32 for Cyclist, and 8.45/8.31/7.71 for Pedestrian on the local
KITTI validation split (easy/moderate/hard). These are diagnostic results, not
official test-server results.

Before reporting benchmark results, use the upstream MMDetection3D KITTI metric
logic (or a regression-tested CPU overlap backend with identical criterion
semantics) and submit test-set detections to the official KITTI server.
