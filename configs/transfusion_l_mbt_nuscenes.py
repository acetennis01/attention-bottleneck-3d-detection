"""Candidate MBT training over pretrained TransFusion-L; baseline-gated.

Not launched automatically. The gate requires a successful full-validation
baseline reproduction and an exact LiDAR/head tensor transfer first.
"""
from copy import deepcopy

_base_ = ['./transfusion_l_nuscenes_pretrained.py']
custom_imports = dict(imports=[
    'projects.BEVFusion.bevfusion',
    'projects.myfusion.fusion.transfusion_l_mbt',
    'projects.myfusion.hooks.transfusion_l_initialization',
    'projects.myfusion.hooks.fresh_fusion_training'], allow_failed_imports=False)

input_modality = dict(use_lidar=True, use_camera=True)
# Mandatory initialization hook performs the audited native-backend transfer.
# Resume still restores the full native MBT checkpoint through MMEngine.
load_from = None
model = dict(type='TransFusionLMBTDetector', freeze_lidar=True,
    data_preprocessor=dict(mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375], bgr_to_rgb=True),
    img_backbone=dict(type='mmdet.ResNet', depth=50, num_stages=4,
        out_indices=(1, 2, 3), frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False), norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(type='mmdet.FPN', in_channels=[512, 1024, 2048],
        out_channels=256, num_outs=3), img_feature_level=0,
    camera_aux_loss_weight=.25,
    fusion_module=dict(type='NuScenesCameraFocusedMBTBEVFusionV2',
        in_img_channels=256, in_lidar_channels=512, embed_dim=256,
        num_bottleneck_tokens=8, num_heads=8, num_layers=4, mlp_ratio=4.,
        token_grid_size=(16, 16), attn_dropout=0., proj_dropout=.1,
        bottleneck_init_std=.02, point_cloud_range=_base_.point_cloud_range,
        height_anchors=(-1.5, -.5, .5, 1.5), camera_aux_num_classes=10,
        height_score_limit=10., lidar_spatial_drop_prob=0.,
        use_local_camera_residual=False, camera_aggregation_mode='valid_only'))

# Keep RGB/points/calibration paired. No untracked object pasting or LiDAR-only
# spatial augmentation that would invalidate camera alignment.
image_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='MultiViewWrapper', process_fields=('img', 'cam2img', 'lidar2cam'),
         transforms=[dict(type='RandomResize3D', scale=(800, 450),
             ratio_range=(1., 1.), keep_ratio=True)])]
meta_keys = ['cam2img', 'lidar2cam', 'img_shape', 'img_path', 'sample_idx',
             'num_pts_feats', 'box_type_3d', 'box_mode_3d']
test_pipeline = deepcopy(image_pipeline) + deepcopy(_base_.test_pipeline[:-1]) + [
    dict(type='Pack3DDetInputs', keys=['img', 'points'], meta_keys=meta_keys)]
train_pipeline = deepcopy(image_pipeline) + [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, load_dim=5, use_dim=5,
        pad_empty_sweeps=True, remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='ObjectNameFilter', classes=_base_.class_names),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_bboxes_3d',
        'gt_labels_3d'], meta_keys=meta_keys)]

train_dataloader = dict(_delete_=True, batch_size=1, num_workers=2, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='NuScenesDataset', data_root=_base_.data_root,
        ann_file='nuscenes_infos_train.pkl', data_prefix=_base_.data_prefix,
        metainfo=_base_.metainfo, pipeline=train_pipeline,
        modality=input_modality, use_valid_flag=True, test_mode=False,
        box_type_3d='LiDAR'))
val_dataloader = dict(dataset=dict(modality=input_modality, pipeline=test_pipeline))
test_dataloader = val_dataloader

# Initial hypothesis: same 40k-update budget as the previous fresh MBT model,
# conservative 1e-4 peak LR. It is not claimed to be the optimal schedule.
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=160000,
    val_begin=10000, val_interval=10000)
optim_wrapper = dict(_delete_=True, type='OptimWrapper', accumulative_counts=4,
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=.01),
    clip_grad=dict(max_norm=10., norm_type=2),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=.2)}))
param_scheduler = [
    dict(type='LinearLR', start_factor=.01, begin=0, end=2000, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=158000, eta_min_ratio=.01,
        begin=2000, end=160000, by_epoch=False)]
custom_hooks = [dict(type='TransFusionLInitializationHook',
    reference_checkpoint=_base_.lidar_checkpoint,
    baseline_report='work_dirs/transfusion_l_pretrained_baseline/baseline_verification.json'),
    dict(type='VersionedNuScenesValidationHook')]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000,
        max_keep_ckpts=2, save_last=True,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/NDS', rule='greater'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False))
log_processor = dict(by_epoch=False)
work_dir = 'work_dirs/transfusion_l_mbt_r50_e256_b8_seed0'
val_evaluator = dict(jsonfile_prefix=work_dir + '/validation',
    modality=input_modality)
test_evaluator = val_evaluator
