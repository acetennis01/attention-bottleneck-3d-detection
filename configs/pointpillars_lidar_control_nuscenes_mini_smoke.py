"""One-epoch smoke test for the nuScenes mini LiDAR control."""

_base_ = ['./pointpillars_lidar_control_nuscenes_mini.py']

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=1,
        save_last=True))
work_dir = 'work_dirs/pointpillars_nuscenes_mini_smoke'

