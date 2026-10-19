"""Larger six-camera V2 experiment; independent of historical nuScenes runs."""

_base_ = ['./my_fusion_mbt_bev_nuscenes.py']

custom_imports = dict(
    imports=[
        'projects.myfusion.fusion.mbt_bev_nuscenes_camera_v2',
        'projects.myfusion.fusion.mbt_detector_nuscenes_camera_v2',
        'projects.myfusion.hooks.staged_fusion_training',
        'projects.myfusion.hooks.camera_v2_masking',
        'projects.myfusion.hooks.fusion_diagnostics',
    ], allow_failed_imports=False)

model = dict(
    type='NuScenesCameraFocusedMBTBEVFusionDetectorV2',
    img_backbone=dict(
        _delete_=True, type='mmdet.ResNet', depth=50, num_stages=4,
        out_indices=(1, 2, 3), frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False), norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(type='mmdet.FPN', in_channels=[512, 1024, 2048],
                  out_channels=256, num_outs=3),
    img_feature_level=0,
    camera_aux_loss_weight=0.25,
    fusion_module=dict(
        _delete_=True, type='NuScenesCameraFocusedMBTBEVFusionV2',
        in_img_channels=256, in_lidar_channels=256, embed_dim=256,
        num_bottleneck_tokens=8, num_heads=8, num_layers=4, mlp_ratio=4.0,
        token_grid_size=(16, 16), attn_dropout=0.0, proj_dropout=0.1,
        bottleneck_init_std=0.02, point_cloud_range=_base_.point_cloud_range,
        height_anchors=(-1.5, -0.5, 0.5, 1.5),
        camera_aux_num_classes=10, height_score_limit=10.0,
        lidar_spatial_drop_prob=0.10, use_local_camera_residual=False,
        camera_aggregation_mode='valid_only'))

# Retain the nuScenes split, six views, sweeps, range, classes, effective
# batch size and 24-epoch schedule. Use V2's staged joint fine-tuning, not
# historical nuScenes' permanently frozen LiDAR and camera backbones.
train_dataloader = dict(batch_size=1, num_workers=2)
optim_wrapper = dict(
    accumulative_counts=4,
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
        }))
custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=5),
    dict(type='CameraV2MaskingHook', max_drop_prob=0.10,
         warmup_epochs=2, ramp_epochs=3),
    dict(type='FusionDiagnosticsHook', interval=500),
]
resume = False
work_dir = 'work_dirs/nuscenes_camera_v2_r50_e256_b8_seed0'
val_evaluator = dict(jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
