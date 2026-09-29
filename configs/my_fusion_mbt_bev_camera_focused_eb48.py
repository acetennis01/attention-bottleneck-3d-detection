"""Camera-focused, bottleneck-mediated KITTI fusion experiment.

The model warm-starts from the matched LiDAR-only detector and keeps that
detector fixed while the corrected camera/fusion path is trained. A camera-
only BEV center loss supervises aligned visual tokens before fusion. All
camera-to-LiDAR semantic communication remains MBT-mediated.
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
    img_backbone=dict(
        # Preserve stable ImageNet features while the fusion path learns.
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True),
    camera_aux_loss_weight=0.25,
    camera_aux_focal_alpha=0.25,
    camera_aux_focal_gamma=2.0,
    fusion_module=dict(
        # Disable the direct local image residual: cross-modal semantics must
        # pass through the shared attention bottleneck tokens.
        use_local_camera_residual=False,
        lidar_token_drop_prob=0.0,
        lidar_bev_drop_prob=0.0,
        camera_aux_num_classes=3,
        height_score_limit=10.0,
        fusion_residual_max_scale=0.1,
        fusion_residual_init_scale=0.02,
    ),
)

# Warm-start the exact LiDAR encoder and Anchor3DHead used by the control.
load_from = (
    'work_dirs/pointpillars_lidar_control_eb48_stable_momentum_seed0/'
    'best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_'
    'epoch_28.pth')

epoch_num = 40
# Keep the transferred detector fixed so improvements are attributable to the
# corrected camera/fusion path rather than LiDAR-only representation drift.
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=epoch_num),
    dict(type='FusionDiagnosticsHook', interval=100),
]

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=2)
optim_wrapper = dict(
    accumulative_counts=8,
    optimizer=dict(
        type='AdamW',
        lr=3e-4,
        betas=(0.95, 0.99),
        weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'fusion_module': dict(lr_mult=1.0),
            'voxel_encoder': dict(lr_mult=0.05),
            'middle_encoder': dict(lr_mult=0.05),
            'backbone': dict(lr_mult=0.05),
            'neck': dict(lr_mult=0.05),
            'bbox_head': dict(lr_mult=0.05),
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
