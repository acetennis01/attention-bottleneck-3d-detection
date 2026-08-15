"""MBT rerun with the intended effective batch size of 48.

The upstream ``8xb6`` PointPillars recipe applies AdamW once per 48 samples.
On one GPU, batch size 6 with eight-way gradient accumulation preserves that
optimizer-update frequency without changing the model, data, or LR schedule.
"""

_base_ = ['./my_fusion_mbt_bev.py']

custom_imports = dict(
    imports=[
        'projects.myfusion.fusion.mbt_bev',
        'projects.myfusion.fusion.mbt_detector',
        'projects.myfusion.hooks.fusion_diagnostics',
    ],
    allow_failed_imports=False,
)

train_dataloader = dict(batch_size=6)
optim_wrapper = dict(accumulative_counts=8)
custom_hooks = [dict(type='FusionDiagnosticsHook', interval=100)]

# Correct the migrated cyclic-momentum values.  CosineAnnealingMomentum's
# ``eta_min`` is an absolute momentum, not a ratio.  The inherited values
# 0.85/0.95 and 1.0 otherwise drive AdamW beta1 to 1.0 at the final update.
lr = 1e-3
epoch_num = 80
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=0.95,
        begin=epoch_num * 0.4,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
]

work_dir = 'work_dirs/mbt_bev_eb48_stable_momentum_seed0'
val_evaluator = dict(pklfile_prefix=work_dir)
test_evaluator = dict(pklfile_prefix=work_dir)
