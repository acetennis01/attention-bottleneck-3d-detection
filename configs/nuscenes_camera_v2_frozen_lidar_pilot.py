"""Two additional epochs from MBT epoch 5, with the entire LiDAR path fixed.

Model-only warm start with a fresh optimizer. Pilot epochs 1/2 are additional
epochs after the selected original epoch 5; they are not a resume of its
optimizer, scheduler, or epoch counter. Do not extend without validation review.
"""

_base_ = ['./nuscenes_camera_v2_no_mask.py']

custom_imports = dict(
    imports=_base_.custom_imports.imports + [
        'projects.myfusion.hooks.frozen_lidar_integrity'],
    allow_failed_imports=False)

reference_checkpoint = (
    'work_dirs/nuscenes_camera_v2_r50_e256_b8_nomask_seed0/'
    'best_NuScenes metric_pred_instances_3d_NuScenes_NDS_epoch_5.pth')
load_from = reference_checkpoint
resume = False

# More than the bounded two-epoch pilot, also safe for a later explicitly
# approved continuation within the original 24-epoch experiment horizon.
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=24),
    dict(type='CameraV2MaskingHook', max_drop_prob=0.0,
         warmup_epochs=2, ramp_epochs=3),
    dict(type='FusionDiagnosticsHook', interval=500),
    dict(type='FrozenLiDARIntegrityHook',
         reference_checkpoint=reference_checkpoint),
]
train_cfg = dict(max_epochs=2, val_begin=1, val_interval=1)

# Keep the previous AdamW base LR, multipliers, and gentle 24-epoch cosine
# schedule. Truncate after two epochs; do not compress the cosine to two.
# No separate learning-rate ablation or architecture change is introduced.
work_dir = 'work_dirs/nuscenes_camera_v2_r50_e256_b8_frozen_e5_pilot_seed0'
val_evaluator = dict(jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
