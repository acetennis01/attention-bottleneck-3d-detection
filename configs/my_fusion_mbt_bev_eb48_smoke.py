"""Two-update memory and gradient smoke test for the EB48 MBT run."""

_base_ = ['./my_fusion_mbt_bev_eb48.py']

# Sixteen microbatches produce two effective-batch-48 updates.  The second is
# required to verify upstream fusion gradients after the zero-initialized BEV
# residual adapter has received its first update.
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

work_dir = 'work_dirs/mbt_bev_eb48_smoke_two_updates'
