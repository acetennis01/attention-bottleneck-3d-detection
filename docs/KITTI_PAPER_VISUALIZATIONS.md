# Updated KITTI paper figures

Generated on 2026-11-02 with the larger ResNet-50/FPN, 256-D, eight-token
ABFuse model's selected epoch-14 checkpoint. These replace the historical
four-token model's illustrations, not its reported experimental results.

Local outputs: `work_dirs/kitti_paper_figures_20261102_b8/`.
The same directory exists under `/home/yab/mmdetection3d/` on Haddr's PC.
Historical images under `docs/` and Downloads are unchanged. The selected
new paper figures are published separately under
[`docs/figures/kitti_v2`](figures/kitti_v2/README.md).

The selected attention figure is **Alternative 1**, using new frames 005472
(Pedestrian), 001707 (Cyclist) and 007462 (Car), not the initial three IDs below.
All use the same larger eight-token epoch-14 checkpoint. See the figure-folder
provenance for exact source hashes and class-matched 3D IoUs.

- `qualitative_midpoint_examples.png`: KITTI validation frames 000008, 000343,
  and 000767, selected for road visibility before new inference. Despite the
  retained filename, these are validation frames, not raw-video midpoints.
  Both LiDAR-only and ABFuse use camera-FOV reduced points and their configured
  point-cloud range. The full 360-degree scan is displayed for reference only.
  The same 0.30 confidence threshold is applied to both models. GT boxes are
  restricted to the three evaluated classes and the shown BEV range.
- `attention_maps_camera_bev.png`: previous sample IDs 006804 (Pedestrian),
  002206 (Cyclist), and 000767 (Car), recomputed with the newer checkpoint.
  Selected detections match moderate GT using class-specific 3D IoU thresholds
  (0.5 for Pedestrian/Cyclist, 0.7 for Car). Their matched IoUs are recorded.
- `attention_*.npz`: raw captured attention weights, height-sampling weights,
  and normalized spatial attention for reproducibility.
- `visualization_manifest.json`: exact checkpoints/configs, SHA-256 digests,
  selected objects, output hashes, and inference verification results.

The attention visualization multiplies the head-averaged final-layer
LiDAR-query-to-bottleneck weights by the bottleneck-query-to-camera weights.
It is an illustrative final-layer attention path, **not** a full multi-layer
rollout or causal attribution. Camera-visible cells are renormalized; learned
height weights/calibration project the map onto the camera image. Each heatmap
is independently normalized and smoothed for display. A highlighted object
need not coincide with the strongest attention region.

Both checkpoints passed strict state-dictionary loading and matched their
previous full-validation checkpoint hashes. Attention capture matched ordinary
inference on all five unique frames (largest score difference below 1e-6).
LiDAR training masks were disabled. Box footprints use the bottom-face corners
`[0, 3, 7, 4]` in MMDetection3D's LiDAR corner convention.

## Reproduce on Haddr

From `/home/yab/mmdetection3d` in the `openmmlab` environment:

```bash
export PYTHONPATH="$PWD"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
  projects/myfusion/tools/render_kitti_paper_figures.py \
  --fusion-report work_dirs/camera_ablations_kitti_nuscenes_20261025/kitti_v2_b8/correct/attempt_01/metrics.json \
  --baseline-report work_dirs/kitti_camera_ablations_20261018_v2b8_vs_legacyb8/lidar/correct/metrics.json \
  --output-dir work_dirs/kitti_paper_figures_NEW_RUN
```

Use a new output directory. No training, evaluation metrics, model parameters,
or original dataset files are modified.

## Paper captions

Detection figure:

```latex
\caption{Qualitative KITTI validation predictions from the larger
eight-token ABFuse model. Top panels compare the 360-degree LiDAR
reference, LiDAR-only predictions, ABFuse predictions, and ground truth.
The camera image shows projected ABFuse detections. Both models use
camera-FOV LiDAR input and a confidence threshold of 0.30.}
```

Attention figure (replace the old reference to four bottleneck tokens):

```latex
\caption{Final-layer attention paths from a selected LiDAR query
through eight bottleneck tokens to camera tokens in the larger ABFuse
model. Left: attention projected onto the camera image. Right: the
corresponding BEV map with LiDAR points and predictions. White outlines
mark the matched ground-truth object and detection. Each heatmap is
normalized independently for visualization.}
```
