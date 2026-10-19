"""Eight iterations, two accumulated updates; execution check, not AP."""

_base_ = ['./nuscenes_camera_v2.py']

train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop', max_iters=8, val_interval=1000)
val_cfg = None
val_dataloader = None
val_evaluator = None
param_scheduler = []
train_dataloader = dict(num_workers=0, persistent_workers=False)
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=5),
    dict(type='FusionDiagnosticsHook', interval=4),
]
default_hooks = dict(
    logger=dict(interval=4),
    checkpoint=dict(
        _delete_=True, type='CheckpointHook', by_epoch=False, interval=8,
        max_keep_ckpts=1, save_last=True))
work_dir = 'work_dirs/nuscenes_camera_v2_smoke'
