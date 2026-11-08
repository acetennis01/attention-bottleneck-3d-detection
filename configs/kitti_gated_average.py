"""Matched average-bottleneck control without camera dropout."""
_base_ = ['./kitti_gated_b8.py']
model = dict(fusion_module=dict(
    camera_drop_prob=0.0, use_bottleneck_gate=False))
work_dir = 'work_dirs/kitti_gated_average_r50_b8_seed0'
val_evaluator = dict(pklfile_prefix=work_dir + '/val_predictions')
test_evaluator = dict(pklfile_prefix=work_dir + '/test_predictions')
