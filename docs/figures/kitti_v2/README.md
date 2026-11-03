# Selected larger KITTI model figures

These are fresh inference outputs from the larger eight-token ABFuse model,
ResNet-50/FPN, selected epoch 14. They are not renamed original-model images.

- [Detection comparison](detection_comparison.png): KITTI validation frames
  000008, 000343 and 000767. Same LiDAR-only baseline, calibration and confidence
  threshold (0.30); full 360-degree scans are reference-only.
- [Attention maps — selected Alternative 1](attention_maps_camera_bev.png):
  Pedestrian 005472, Cyclist 001707, Car 007462. Objects are class-matched 3D
  true positives (IoU thresholds 0.5/0.5/0.7). White/dark outlines identify
  GT on the camera image and the matched prediction in BEV; colored boxes
  are other predictions. Scene selection is for illustration, not representative
  sampling or quantitative evidence of object-localized attention.
- `provenance.json`: source/output hashes and selected checkpoint identity.
- `detection_source_manifest.json`, `attention_source_manifest.json`: original
  rendering records; paths to extra alternative figures refer to local/GPU-host
  artifacts, not all of them are republished here.

Use **eight**, not four, tokens in the attention caption. Maps show independently
normalized final-layer attention paths through the bottlenecks, not full
multi-layer attention rollout or causal pixel attribution. Colors across
examples should not be compared numerically. Attention can peak on context.

Reproduction code: `tools/render_kitti_paper_figures.py` and
`tools/render_kitti_attention_alternatives.py`. Full commands and interpretation:
[visualization guide](../../KITTI_PAPER_VISUALIZATIONS.md).
