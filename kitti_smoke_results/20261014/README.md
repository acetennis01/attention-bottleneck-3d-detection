# KITTI bottleneck smoke results — 2026-10-14

All five token counts passed training and subset validation on Haddr's RTX 4090.
The original four-token checkpoint also passed strict loading and subset
inference/evaluation. These runs test execution, not detection quality.

| Bottleneck tokens | Training / validation | Training peak allocated memory (MiB) |
|---|---|---:|
| 1 | Passed | 3013.8 |
| 2 | Passed | 3014.0 |
| 4 | Passed | 3015.7 |
| 8 | Passed | 3017.3 |
| 16 | Passed | 3021.6 |

Every training test used 96 KITTI training frames, physical batch size six,
eight-step accumulation, 16 iterations, and two optimizer updates. Validation
used 32 frames. Each test then performed one backward pass with the LiDAR
detector unfrozen, without a further optimizer update. The memory figures above
cover the short frozen-LiDAR training phase; they are not full-training memory
benchmarks. AP from these subsets must not be reported as benchmark performance.

Verified in every variant:

- Exact initialization of all LiDAR detector state from the epoch-28 baseline.
- Finite losses, features, gradients, and validation metrics.
- Calibrated camera cells visible to the fusion module.
- Actual parameter updates in the image backbone, camera projection,
  bottleneck tokens, and BEV residual projection.
- No changes to frozen LiDAR parameters or BatchNorm statistics.
- Nonzero finite gradients in the LiDAR backbone and detection head when unfrozen.
- Successful checkpoint serialization.

The original four-token KITTI checkpoint requires the historical fusion module
from commit `c2dad89`. Later nuScenes changes are incompatible with it. The
isolated `KittiV1SymmetricMBTBEVFusion` module loads all original checkpoint
parameters strictly. See `docs/KITTI_BOTTLENECK_SMOKE.md` for implementation and
recipe details. In particular, the original KITTI recipe used 40 epochs and a
five-epoch LiDAR freeze, followed by joint training.

Remote output directory:

```text
/home/yab/mmdetection3d/work_dirs/kitti_bottleneck_smoke_kpJcL2/
```

Local subdirectories include each `smoke_report.json`, resolved smoke config,
training log, and metric log. The full bottleneck ablation has not been started.
