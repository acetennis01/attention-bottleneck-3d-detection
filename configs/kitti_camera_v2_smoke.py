"""Bounded execution smoke: 16 iterations / two accumulated updates, no AP."""

_base_ = ['./kitti_camera_v2.py']

train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop', max_iters=16, val_interval=1000)
val_cfg = None
val_dataloader = None
val_evaluator = None
param_scheduler = []
train_dataloader = dict(num_workers=2, persistent_workers=False)
optim_wrapper = dict(accumulative_counts=8)
# Exercise the actual 0.10 mask immediately in this bounded smoke, instead of
# spending its two optimizer steps in the production no-mask warm-up.
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=5),
    dict(type='FusionDiagnosticsHook', interval=8),
]
default_hooks = dict(
    logger=dict(interval=8),
    checkpoint=dict(
        _delete_=True, type='CheckpointHook', by_epoch=False, interval=16,
        max_keep_ckpts=1, save_last=True))
work_dir = 'work_dirs/kitti_camera_v2_smoke'
