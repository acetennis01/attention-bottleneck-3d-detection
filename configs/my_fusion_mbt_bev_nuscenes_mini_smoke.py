"""One-epoch smoke test for six-camera nuScenes MBT fusion."""

_base_ = ['./my_fusion_mbt_bev_nuscenes_mini.py']

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=1,
        save_last=True))
work_dir = 'work_dirs/mbt_bev_nuscenes_mini_smoke'
val_evaluator = dict(
    jsonfile_prefix='work_dirs/mbt_bev_nuscenes_mini_smoke/results')
test_evaluator = val_evaluator
