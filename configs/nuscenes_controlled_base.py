"""Bounded, model-only continuation controls: 4,000 batches / 1,000 updates.

All arms share the epoch-5 model, a fresh AdamW optimizer, constant learning
rate, seed, sample order, effective batch size and full validation endpoint.
These are short diagnostic continuations, not complete new benchmark runs.
"""

_base_ = ['./nuscenes_camera_v2_no_mask.py']

custom_imports = dict(
    imports=_base_.custom_imports.imports + [
        'projects.myfusion.fusion.mbt_nuscenes_multiscale',
        'projects.myfusion.hooks.controlled_retraining'],
    allow_failed_imports=False)

reference_checkpoint = (
    'work_dirs/nuscenes_camera_v2_r50_e256_b8_nomask_seed0/'
    'best_NuScenes metric_pred_instances_3d_NuScenes_NDS_epoch_5.pth')
load_from = reference_checkpoint
resume = False
randomness = dict(seed=0, deterministic=False, diff_rank_seed=False)
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=4000,
                 val_begin=4000, val_interval=4000)
# No epoch-dependent scheduler restart or compressed cosine confound.
param_scheduler = []
optim_wrapper = dict(optimizer=dict(lr=3e-4), accumulative_counts=4)
default_hooks = dict(
    checkpoint=dict(_delete_=True, type='CheckpointHook', by_epoch=False,
                    interval=1000, max_keep_ckpts=2, save_last=True,
                    save_best='NuScenes metric/pred_instances_3d_NuScenes/NDS',
                    rule='greater'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False))
log_processor = dict(by_epoch=False)

# Arm configs replace this list explicitly, avoiding list-merge ambiguity.
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=24),
    dict(type='FusionDiagnosticsHook', interval=500),
    dict(type='ControlledRetrainingHook', reference_checkpoint=reference_checkpoint,
         frozen_lidar=True),
]

work_dir = 'work_dirs/nuscenes_controlled_20261021/frozen'
val_evaluator = dict(jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
