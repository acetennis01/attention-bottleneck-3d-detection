"""Two-update GPU smoke test for camera-focused MBT training."""

_base_ = ['./my_fusion_mbt_bev_camera_focused_eb48.py']

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=16,
    val_interval=1000,
)
val_cfg = None
val_dataloader = None
val_evaluator = None
param_scheduler = []
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=5),
    dict(type='FusionDiagnosticsHook', interval=8),
]
default_hooks = dict(
    logger=dict(interval=8),
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        by_epoch=False,
        interval=16,
        max_keep_ckpts=1,
        save_last=True))

work_dir = 'work_dirs/mbt_bev_camera_focused_eb48_smoke'
