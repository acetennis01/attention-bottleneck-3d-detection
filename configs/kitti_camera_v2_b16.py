"""Controlled V2 token-count experiment: sixteen instead of eight tokens."""

_base_ = ['./kitti_camera_v2.py']

model = dict(fusion_module=dict(num_bottleneck_tokens=16))

work_dir = 'work_dirs/kitti_camera_v2_r50_e256_b16_seed0'
val_evaluator = dict(pklfile_prefix=work_dir + '/val_predictions')
test_evaluator = dict(pklfile_prefix=work_dir + '/test_predictions')
