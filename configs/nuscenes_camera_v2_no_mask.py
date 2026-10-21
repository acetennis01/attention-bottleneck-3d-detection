"""Fresh eight-token stability control: no hard LiDAR feature/token masking.

Retains the camera-only auxiliary task and all bottleneck-mediated exchange.
The interrupted masked experiment/config is preserved for reproducibility.
"""

_base_ = ['./nuscenes_camera_v2.py']

model = dict(fusion_module=dict(lidar_spatial_drop_prob=0.0))
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=5),
    # Keep this explicitly zero, including when loading a later epoch.
    dict(type='CameraV2MaskingHook', max_drop_prob=0.0,
         warmup_epochs=2, ramp_epochs=3),
    dict(type='FusionDiagnosticsHook', interval=500),
]
resume = False
work_dir = 'work_dirs/nuscenes_camera_v2_r50_e256_b8_nomask_seed0'
val_evaluator = dict(jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
