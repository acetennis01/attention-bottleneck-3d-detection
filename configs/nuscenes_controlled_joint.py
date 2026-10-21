"""One change versus frozen: train LiDAR/head, including BN statistics."""
_base_ = ['./nuscenes_controlled_base.py']
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=0),
    dict(type='FusionDiagnosticsHook', interval=500),
    dict(type='ControlledRetrainingHook',
         reference_checkpoint=_base_.reference_checkpoint, frozen_lidar=False),
]
work_dir = 'work_dirs/nuscenes_controlled_20261021/joint'
val_evaluator = dict(jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
