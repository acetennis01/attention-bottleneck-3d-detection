"""Train a separate V2 control to measure the effect of coupled masking."""

_base_ = ['./kitti_camera_v2.py']

model = dict(fusion_module=dict(lidar_spatial_drop_prob=0.0))
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=5),
    dict(type='CameraV2MaskingHook', max_drop_prob=0.0),
    dict(type='FusionDiagnosticsHook', interval=100),
]
work_dir = 'work_dirs/kitti_camera_v2_no_mask_seed0'
val_evaluator = dict(pklfile_prefix=work_dir + '/val_predictions')
test_evaluator = dict(pklfile_prefix=work_dir + '/test_predictions')
