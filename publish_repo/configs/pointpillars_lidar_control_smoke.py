"""Two-epoch checkpointed diagnostic for the LiDAR-only control."""

_base_ = ['./pointpillars_lidar_control.py']

work_dir = 'work_dirs/pointpillars_lidar_diagnostic'
train_cfg = dict(by_epoch=True, max_epochs=2, val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=2, save_last=True))
val_evaluator = dict(pklfile_prefix=work_dir)
test_evaluator = dict(pklfile_prefix=work_dir)
