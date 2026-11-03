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

These are **validation-split** results, not KITTI test-server submissions.
KITTI AP40 is averaged over Car, Pedestrian and Cyclist. nuScenes mAP/NDS are
shown as percentages. Every pair below uses its matched LiDAR architecture.

| Dataset / experiment | LiDAR-only | ABFuse | Difference |
|---|---:|---:|---:|
| KITTI, original four-token, moderate 3D AP40 | 38.69 | 42.67 | +3.98 |
| KITTI, larger eight-token, moderate 3D AP40 | 38.69 | **44.39** | **+5.71** |
| KITTI, larger eight-token, moderate BEV AP40 | 52.89 | **55.05** | **+2.17** |
| nuScenes, original PointPillars, mAP / NDS | 25.61 / 44.50 | 27.78 / 45.29 | +2.17 / +0.79 |
| nuScenes, multi-scale PointPillars, mAP / NDS | 25.61 / 44.50 | 30.17 / 46.21 | +4.55 / +1.71 |
| nuScenes, TransFusion-L / Swin-Tiny, mAP / NDS | 64.43 / 69.24 | **64.79 / 69.39** | **+0.36 / +0.15** |

Differences are calculated **before rounding**. KITTI LiDAR/head parameters are
fine-tuned after five frozen epochs; therefore the entire baseline gain cannot
be attributed solely to camera information. The completed nuScenes models keep
the transferred LiDAR/head fixed. Comparisons between different fusion variants
are not controlled architectural ablations.

**Start with [verified results and provenance](results/README.md)** for exact
scores, per-class reports, the completed 1/2/4/8/16-token study, all 30 camera
ablations, checkpoint hashes and saved training configs. Corruption gaps support
camera dependence, not statistical significance or causal attribution.

## Choose an experiment

Paths below are relative to this project. Run commands from the MMDetection3D
root, with the project installed at `projects/myfusion`.

| Experiment | Entry config | Details |
|---|---|---|
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

# Larger KITTI eight-token model; requires the matched LiDAR checkpoint
# at the load_from path inherited from kitti_bottleneck_base.py.
python tools/train.py projects/myfusion/configs/kitti_camera_v2.py
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
