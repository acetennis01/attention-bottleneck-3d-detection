# ABFuse: attention-bottleneck camera–LiDAR 3D detection

An MMDetection3D project for geometrically aligned camera–LiDAR fusion on
**KITTI** and **nuScenes**. All completed model families are available on
`main`; switching experimental branches is no longer necessary.

Camera features are aligned with LiDAR BEV locations using calibration. Separate
transformer branches exchange information through shared attention-bottleneck
tokens. LiDAR queries read the bottlenecks and update BEV features before
3D detection. `MBT` in Python class/file names refers to this mechanism; the
paper's model name is **ABFuse**. Class names are retained for checkpoint compatibility.

## Results at a glance

The current reference is **gated ABFuse**. KITTI uses ResNet-50/FPN,
PointPillars, width 256 and eight bottleneck tokens; nuScenes uses
nuImages Swin-Tiny and TransFusion-L. Both keep the LiDAR detector frozen.
These are validation results; KITTI overall AP40 averages Car, Pedestrian
and Cyclist, and nuScenes mAP/NDS are percentages.

| Gated reference, clean inputs | LiDAR-only | Gated ABFuse | Gain (points) |
|---|---:|---:|---:|
| KITTI moderate 3D AP40 | 38.6872 | 42.0490 | +3.3618 |
| KITTI moderate BEV AP40 | 52.8851 | 54.6942 | +1.8091 |
| nuScenes mAP / NDS | 64.43 / 69.24 | 64.79 / 69.40 | +0.36 / +0.17 |

- [Gated bottleneck-count, gate/dropout and width results](results/gated_studies_20261113/README.md).
- [Matched KITTI architecture controls](results/part_b_20261110/README.md).
- [nuScenes camera ablations](results/part_b_20261109/README.md).
- [Sensor-corruption evaluations, reliability pilots and latency](results/sensor_reliability_20261120/README.md).
- [Exact results, historical averaged models and provenance](results/README.md).

Explicit camera absence bypasses fusion using a supplied availability flag;
black or shuffled images do not automatically trigger that path. Corruptions
can underperform LiDAR-only. Learned reliability is a separate exploratory
pilot with mixed gains, not a replacement for the reference checkpoint.
The KITTI width study changes fusion width, not the camera backbone;
completed gated multi-scale nuScenes results are not claimed.

## Choose an experiment

Paths below are relative to this project. Run commands from the MMDetection3D
root, with the project installed at `projects/myfusion`.

| Experiment | Entry config | Details |
|---|---|---|
| KITTI gated reference, ResNet-50/FPN, B8 | `configs/kitti_gated_b8.py` | [Matched studies](docs/GATED_STUDIES_20261113.md), [completed results](results/gated_studies_20261113/README.md) |
| nuScenes gated TransFusion-L / Swin-Tiny | `configs/transfusion_l_gated_swint_nuscenes.py` | [Completed comparison](results/part_b_20261109/README.md) |
| KITTI learned-reliability pilots and latency | Generated matched configs | [Reproduction guide](docs/SENSOR_RELIABILITY.md), [results](results/sensor_reliability_20261120/README.md) |
| KITTI LiDAR-only control | `configs/pointpillars_lidar_control_eb48.py` | [Historical reference](EVALUATION.md) |
| KITTI controlled token sweep, ResNet-18 | `configs/kitti_bottleneck_b{1,2,4,8,16}.py` | [Protocol](docs/KITTI_BOTTLENECK_SMOKE.md), [completed results](docs/KITTI_BOTTLENECK_RESULTS.md) |
| KITTI larger, ResNet-50/FPN, eight tokens | `configs/kitti_camera_v2.py` | [Architecture and training](CAMERA_FOCUSED_V2.md) |
| KITTI larger, sixteen tokens | `configs/kitti_camera_v2_b16.py` | [Same controlled V2 recipe](CAMERA_FOCUSED_V2.md) |
| nuScenes PointPillars LiDAR-only | `configs/pointpillars_lidar_control_nuscenes.py` | [Protocol](docs/NUSCENES_TRAINING.md) |
| nuScenes original four-token / ResNet-18 | `configs/my_fusion_mbt_bev_nuscenes.py` | [Protocol](docs/NUSCENES_TRAINING.md) |
| nuScenes multi-scale / ResNet-50, eight tokens | `configs/nuscenes_multiscale_fresh.py` | [Architecture and training](docs/NUSCENES_MULTISCALE_FRESH.md) |
| nuScenes pretrained TransFusion-L baseline | `configs/transfusion_l_nuscenes_pretrained.py` | [Checkpoint verification](docs/TRANSFUSION_L_MBT.md) |
| nuScenes TransFusion-L / nuImages Swin-Tiny | `configs/transfusion_l_mbt_swint_nuscenes.py` | [Training](docs/TRANSFUSION_L_TRAINING.md), [camera initializer](docs/NUIMAGES_SWIN_CAMERA.md) |

The original KITTI epoch-20 result uses a historical implementation. The current
`my_fusion_mbt_bev_camera_focused_eb48.py` alias was subsequently changed for
stable fusion and is **not** that historical trained architecture. Preserve the
versioned `mbt_bev_kitti_v1.py`; use the saved training config/checkpoint metadata
and [compatibility checks](docs/KITTI_BOTTLENECK_SMOKE.md) for old checkpoints.
Do not replace the original paper result with the separate corrected B4 sweep.

## Setup and run

This is a project extension, **not a fork or replacement of MMDetection3D**.
The verified training environment used Python 3.10, PyTorch 2.1.0+cu121,
MMCV 2.1.0, MMDetection 3.2.0 and MMDetection3D 1.4.0. CUDA/compiled ops and
dataset preparation are required; weights and datasets are not stored here.

```bash
cd ~/mmdetection3d
conda activate openmmlab
export PYTHONPATH="$PWD"

# Gated KITTI reference; requires the matched LiDAR initializer.
python tools/train.py projects/myfusion/configs/kitti_gated_b8.py
```

Setup helpers are in `tools/setup_kitti_haddr.sh`, `tools/setup_nuscenes_trainval.sh`,
`tools/setup_transfusion_l.sh` and `tools/setup_nuimages_camera.sh`. Read each
experiment's guide first: ranges, sweeps, channels and initializers differ.
Public TransFusion-L/Swin component checkpoints are verified by pinned hashes;
they are not a complete pretrained ABFuse detector or a reproduction of the
complete BEVFusion architecture.

```bash
# Regressions in the installed GPU environment.
python -m pytest projects/myfusion/tests -q \
  --ignore=projects/myfusion/tests/test_kitti_overlap_backend.py
```

The optional KITTI rotated-IoU backend has separate installation/regression
instructions in [the KITTI audit](docs/KITTI_ARCHITECTURE_AUDIT.md). Smoke tests
and AP on tiny subsets are execution checks, never benchmark results.

## Camera ablations and figures

The published detection/attention figures use the historical averaged R50/B8
epoch-14 model, rather than the gated epoch-18 reference.

- [Completed 30-case camera-ablation suite](docs/CAMERA_ABLATION_SUITE.md):
  correct, black-pixel and shuffled-camera controls at the same selected weights.
- [Updated KITTI figures and reproduction commands](docs/KITTI_PAPER_VISUALIZATIONS.md).
- [Detection comparison](docs/figures/kitti_v2/detection_comparison.png) and
  [selected alternative attention maps](docs/figures/kitti_v2/attention_maps_camera_bev.png),
  freshly rendered with the larger eight-token epoch-14 model and labelled ABFuse.

Attention maps show independently normalized final-layer attention pathways,
not full multi-layer rollouts or causal explanations. Historical four-token
figures and the original KITTI evaluation remain clearly separated.

## Repository map

```text
configs/          Training, baseline, ablation and smoke configs (inheritance retained)
fusion/           Versioned symmetric bottleneck modules and detector integrations
hooks/            Freezing, masking, checkpoint integrity and diagnostics
transforms/       Camera corruption controls
tools/            Setup, smoke, evaluation, publication and visualization scripts
tests/            Model, configuration, corruption and published-result checks
results/          Compact verified metrics, exact saved configs and provenance
docs/             Experiment guides, audits and selected figures
```

[Experiment/config guide](docs/EXPERIMENT_INDEX.md) distinguishes completed paper
models from historical pilots and smoke-only variants. [Repository maintenance](docs/REPOSITORY_MAINTENANCE.md)
explains what is published and what stays local. Historical raw logs/videos and
duplicate smoke artifacts are removed only from Git tracking, not from local
storage or earlier commits. No history rewriting or branch deletion is required.
