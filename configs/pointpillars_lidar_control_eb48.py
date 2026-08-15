"""Matched LiDAR control using the intended effective batch size of 48."""

_base_ = ['./pointpillars_lidar_control.py']

train_dataloader = dict(batch_size=6)
optim_wrapper = dict(accumulative_counts=8)

# Correct the inherited migrated momentum schedule.  These are absolute
# AdamW beta1 targets: descend from 0.95 to 0.85, then return to 0.95.
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

work_dir = 'work_dirs/pointpillars_lidar_control_eb48_stable_momentum_seed0'
val_evaluator = dict(pklfile_prefix=work_dir)
test_evaluator = dict(pklfile_prefix=work_dir)
