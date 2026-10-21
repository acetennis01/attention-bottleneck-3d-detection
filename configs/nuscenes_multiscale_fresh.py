"""Fresh camera/MBT optimization over a verified, permanently fixed LiDAR model.

Not an epoch-5 MBT continuation and not random initialization of every backbone.
40,000 optimizer updates = 160,000 microbatches at physical batch 1/accumulation 4.
This initial full-run schedule remains a hypothesis, not a validated optimum.
"""

_base_ = ['./nuscenes_camera_v2_no_mask.py']

custom_imports = dict(
    imports=_base_.custom_imports.imports + [
        'projects.myfusion.fusion.mbt_nuscenes_multiscale',
        'projects.myfusion.hooks.fresh_fusion_training'],
    allow_failed_imports=False)

lidar_checkpoint = (
    'work_dirs/pointpillars_nuscenes/'
    'best_NuScenes metric_pred_instances_3d_NuScenes_NDS_epoch_24.pth')
lidar_checkpoint_sha256 = 'af4443ce5eb32249fe2454d141aade6b5db93be8881169d93c65ab669aa7dfb2'
load_from = lidar_checkpoint
resume = False
model = dict(type='NuScenesMultiScaleMBTDetectorV2',
             camera_aux_loss_weight=0.25,
             fusion_module=dict(type='NuScenesMultiScaleMBTV2',
                                lidar_spatial_drop_prob=0.0))

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=160000,
                 val_begin=10000, val_interval=10000)
optim_wrapper = dict(accumulative_counts=4, optimizer=dict(lr=3e-4))
# Scheduler times are microbatches, with boundaries divisible by accumulation.
# Warm up over 500 optimizer updates, then decay over the remaining budget.
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, begin=0, end=2000, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=158000, eta_min_ratio=0.01,
         begin=2000, end=160000, by_epoch=False),
]
custom_hooks = [
    # IterBasedTrainLoop keeps epoch=0. Nonzero freeze_epochs means permanent
    # freezing here; no epoch-five transition is present in this recipe.
    dict(type='StagedFusionTrainingHook', freeze_epochs=1),
    dict(type='FusionDiagnosticsHook', interval=500),
    dict(type='FreshFusionTrainingHook', reference_checkpoint=lidar_checkpoint,
         reference_sha256=lidar_checkpoint_sha256, interval=1000),
    dict(type='VersionedNuScenesValidationHook'),
]
default_hooks = dict(
    checkpoint=dict(_delete_=True, type='CheckpointHook', by_epoch=False,
                    interval=2000, max_keep_ckpts=2, save_last=True,
                    save_best='NuScenes metric/pred_instances_3d_NuScenes/NDS',
                    rule='greater'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False))
log_processor = dict(by_epoch=False)
randomness = dict(seed=0, deterministic=False, diff_rank_seed=False)
work_dir = 'work_dirs/nuscenes_multiscale_fresh_r50_e256_b8_seed0'
val_evaluator = dict(jsonfile_prefix=work_dir + '/validation')
test_evaluator = val_evaluator
