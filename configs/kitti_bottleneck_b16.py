"""Sixteen-token KITTI ablation with valid-only camera-height normalization."""

_base_ = ['./kitti_bottleneck_base.py']

model = dict(fusion_module=dict(num_bottleneck_tokens=16))

work_dir = 'work_dirs/kitti_bottleneck_b16_valid_only_seed0'
val_evaluator = dict(pklfile_prefix=work_dir + '/val_predictions')
test_evaluator = dict(pklfile_prefix=work_dir + '/test_predictions')
