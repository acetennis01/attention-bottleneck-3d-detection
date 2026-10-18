"""New KITTI camera-focused MBT experiment; does not replace the token study."""

_base_ = ['./kitti_bottleneck_base.py']

custom_imports = dict(
    imports=[
        'projects.myfusion.fusion.mbt_bev_camera_v2',
        'projects.myfusion.fusion.mbt_detector_camera_v2',
        'projects.myfusion.hooks.staged_fusion_training',
        'projects.myfusion.hooks.camera_v2_masking',
        'projects.myfusion.hooks.fusion_diagnostics',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='CameraFocusedMBTBEVFusionDetectorV2',
    img_backbone=dict(
        _delete_=True,
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        # Fine-tune C3-C5 while preserving pretrained image BN statistics.
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=3),
    img_feature_level=0,  # P3: stride 8, with top-down C4/C5 information.
    camera_aux_loss_weight=0.25,
    fusion_module=dict(
        _delete_=True,
        type='CameraFocusedMBTBEVFusionV2',
        in_img_channels=256,
        in_lidar_channels=384,
        embed_dim=256,
        num_bottleneck_tokens=8,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4.0,
        token_grid_size=(16, 16),
        attn_dropout=0.0,
        proj_dropout=0.1,
        bottleneck_init_std=0.02,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        height_anchors=(-1.5, -0.75, 0.0, 0.75),
        camera_aux_num_classes=3,
        height_score_limit=10.0,
        lidar_spatial_drop_prob=0.10,
        use_local_camera_residual=False,
        camera_aggregation_mode='valid_only',
    ),
)

# Keep both the physical and effective batch sizes of the historical study.
# The GPU smoke also probes the memory-heavier, unfrozen LiDAR backward pass.
train_dataloader = dict(batch_size=6, num_workers=4)
optim_wrapper = dict(
    accumulative_counts=8,
    clip_grad=dict(max_norm=10, norm_type=2),
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys={
            'img_backbone': dict(lr_mult=0.2),
            'img_neck': dict(lr_mult=1.0),
            'fusion_module': dict(lr_mult=1.0),
            'voxel_encoder': dict(lr_mult=0.5),
            'middle_encoder': dict(lr_mult=0.5),
            'backbone': dict(lr_mult=0.5),
            'neck': dict(lr_mult=0.5),
            'bbox_head': dict(lr_mult=0.5),
        }),
)
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=5),
    dict(type='CameraV2MaskingHook', max_drop_prob=0.10,
         warmup_epochs=2, ramp_epochs=3),
    dict(type='FusionDiagnosticsHook', interval=100),
]

# Inherit the LiDAR-only warm start, 40-epoch cosine schedule, seed, KITTI split,
# calibration-safe augmentations, and strict moderate 3D AP40 selection.
# A 128-D MBT checkpoint is NOT compatible; this is a fresh fusion experiment.
resume = False
work_dir = 'work_dirs/kitti_camera_v2_r50_e256_b8_seed0'
val_evaluator = dict(pklfile_prefix=work_dir + '/val_predictions')
test_evaluator = dict(pklfile_prefix=work_dir + '/test_predictions')
