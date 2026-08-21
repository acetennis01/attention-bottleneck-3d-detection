"""Camera-focused, bottleneck-mediated KITTI fusion experiment.

The model warm-starts from the matched LiDAR-only detector. During a five-
epoch fusion warm-up, the pretrained LiDAR encoder and detection head are
frozen. Camera-covered LiDAR tokens and channels are stochastically hidden,
and an auxiliary camera-BEV center loss is applied after bottleneck readout.
All camera-to-LiDAR semantic communication remains MBT-mediated.
"""

_base_ = ['./my_fusion_mbt_bev_aligned_eb48.py']

custom_imports = dict(
    imports=[
        'projects.myfusion.fusion.mbt_bev',
        'projects.myfusion.fusion.mbt_detector',
        'projects.myfusion.hooks.fusion_diagnostics',
        'projects.myfusion.hooks.staged_fusion_training',
    ],
    allow_failed_imports=False,
)

model = dict(
    camera_aux_loss_weight=0.25,
    camera_aux_focal_alpha=0.25,
    camera_aux_focal_gamma=2.0,
    fusion_module=dict(
        # Disable the direct local image residual: cross-modal semantics must
        # pass through the shared attention bottleneck tokens.
        use_local_camera_residual=False,
        lidar_token_drop_prob=0.15,
        lidar_bev_drop_prob=0.10,
        camera_aux_num_classes=3,
    ),
)

# Warm-start the exact LiDAR encoder and Anchor3DHead used by the control.
load_from = (
    'work_dirs/pointpillars_lidar_control_eb48_stable_momentum_seed0/'
    'best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_'
    'epoch_28.pth')

custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=5),
    dict(type='FusionDiagnosticsHook', interval=100),
]

# Fine-tune gently from the pretrained LiDAR solution. Camera/fusion modules
# learn four times faster than the LiDAR path after staged unfreezing.
epoch_num = 40
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=2)
optim_wrapper = dict(
    accumulative_counts=8,
    optimizer=dict(
        type='AdamW',
        lr=3e-4,
        betas=(0.95, 0.99),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=2.0),
            'fusion_module': dict(lr_mult=2.0),
            'voxel_encoder': dict(lr_mult=0.5),
            'middle_encoder': dict(lr_mult=0.5),
            'backbone': dict(lr_mult=0.5),
            'neck': dict(lr_mult=0.5),
            'bbox_head': dict(lr_mult=0.5),
        }),
)
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num,
        eta_min=3e-6,
        begin=0,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
]

work_dir = 'work_dirs/mbt_bev_camera_focused_eb48_seed0'
val_evaluator = dict(pklfile_prefix=work_dir)
test_evaluator = dict(pklfile_prefix=work_dir)

