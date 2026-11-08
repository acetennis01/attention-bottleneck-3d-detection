"""Extra-capacity control: train residual with no camera information.

Bypass is disabled deliberately: otherwise this would be the frozen baseline
with zero detection gradient, not a trainable extra-capacity control. Auxiliary
camera loss is zero through the availability mask in the shared graph.
"""
_base_ = ['./kitti_gated_b8.py']
model = dict(fusion_module=dict(
    camera_mode='missing', camera_drop_prob=0.0, bypass_missing=False))
work_dir = 'work_dirs/kitti_gated_absent_control_r50_b8_seed0'
val_evaluator = dict(pklfile_prefix=work_dir + '/val_predictions')
test_evaluator = dict(pklfile_prefix=work_dir + '/test_predictions')
