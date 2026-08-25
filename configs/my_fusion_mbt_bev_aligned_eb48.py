"""Calibration-aware camera-to-BEV MBT experiment.

This preserves the controlled EB48 training recipe while replacing the
unaligned image token grid with image features sampled at physical KITTI BEV
locations. It must be trained from scratch; old MBT checkpoints do not contain
the alignment parameters.
"""

_base_ = ['./my_fusion_mbt_bev_eb48.py']

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

model = dict(
    fusion_module=dict(
        point_cloud_range=point_cloud_range,
        # Sample ground-to-roof locations in the LiDAR coordinate system.
        height_anchors=(-1.5, -0.75, 0.0, 0.75),
    ))

work_dir = 'work_dirs/mbt_bev_aligned_eb48_seed0'
val_evaluator = dict(pklfile_prefix=work_dir)
test_evaluator = dict(pklfile_prefix=work_dir)

