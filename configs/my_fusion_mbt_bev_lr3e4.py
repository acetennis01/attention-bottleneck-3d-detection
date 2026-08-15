"""Seed-0 MBT stability experiment with a three-times lower cyclic LR.

This config changes only the optimizer/scheduler learning-rate scale relative
to ``my_fusion_mbt_bev.py``.  The model, data split, augmentations, batch size,
seed, validation cadence, and 80-epoch duration remain unchanged.
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

# The inherited PointPillars schedule rises from 1e-3 to 1e-2.  Preserve its
# shape while reducing the complete LR curve by 3.33x.
lr = 3e-4
epoch_num = 80
optim_wrapper = dict(optimizer=dict(lr=lr))
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
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num,
        convert_to_iter_based=True),
]

# Observe modality-specific gradients and the size of the learned BEV update
# without changing the forward pass or loss.
custom_hooks = [dict(type='FusionDiagnosticsHook', interval=500)]

work_dir = 'work_dirs/mbt_bev_lr3e4_seed0'
val_evaluator = dict(pklfile_prefix=work_dir)
test_evaluator = dict(pklfile_prefix=work_dir)

