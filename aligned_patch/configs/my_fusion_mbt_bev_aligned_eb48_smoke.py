"""Two-optimizer-update smoke test for calibration-aware MBT."""

_base_ = ['./my_fusion_mbt_bev_aligned_eb48.py']

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
custom_hooks = [dict(type='FusionDiagnosticsHook', interval=8)]
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        by_epoch=False,
        interval=16,
        max_keep_ckpts=1,
        save_last=True))

work_dir = 'work_dirs/mbt_bev_aligned_eb48_smoke'

