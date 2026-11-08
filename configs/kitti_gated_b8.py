"""Part B prototype: frozen-detector R50/B8, new fusion training from LiDAR.

Not comparable as an isolated gate change to historical jointly trained KITTI.
Use the sibling average, dropout-only and absent-camera controls instead.
"""

_base_ = ['./kitti_camera_v2.py']

custom_imports = dict(
    imports=[
        'projects.myfusion.fusion.mbt_bev_gated',
        'projects.myfusion.fusion.mbt_detector_camera_v2',
        'projects.myfusion.hooks.staged_fusion_training',
        'projects.myfusion.hooks.fusion_diagnostics',
    ], allow_failed_imports=False)

model = dict(fusion_module=dict(
    type='GatedCameraMBTBEVFusion', lidar_spatial_drop_prob=0.0,
    camera_drop_prob=0.2, use_bottleneck_gate=True,
    camera_mode='present', bypass_missing=True))

# Freeze both parameters and normalization buffers for the entire 40 epochs.
# No LiDAR masking: absence fallback must retain the original detector input.
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=40),
    dict(type='FusionDiagnosticsHook', interval=100),
]
work_dir = 'work_dirs/kitti_gated_r50_b8_seed0'
val_evaluator = dict(pklfile_prefix=work_dir + '/val_predictions')
test_evaluator = dict(pklfile_prefix=work_dir + '/test_predictions')
