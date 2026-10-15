"""Four-token control for the valid-only KITTI bottleneck study."""

_base_ = ['./kitti_bottleneck_base.py']

model = dict(fusion_module=dict(num_bottleneck_tokens=4))

work_dir = 'work_dirs/kitti_bottleneck_b4_valid_only_seed0'
val_evaluator = dict(pklfile_prefix=work_dir + '/val_predictions')
test_evaluator = dict(pklfile_prefix=work_dir + '/test_predictions')
