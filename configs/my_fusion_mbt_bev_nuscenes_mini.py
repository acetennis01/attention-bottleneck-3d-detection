"""Six-camera geometry-aligned MBT detector on nuScenes mini."""

_base_ = ['./pointpillars_lidar_control_nuscenes_mini.py']

custom_imports = dict(
    imports=[
        'projects.myfusion.fusion.mbt_bev',
        'projects.myfusion.fusion.mbt_detector',
        'projects.myfusion.fusion.mbt_bev_nuscenes',
        'projects.myfusion.fusion.mbt_detector_nuscenes',
    ],
    allow_failed_imports=False)

input_modality = dict(use_lidar=True, use_camera=True)
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    sweeps='sweeps/LIDAR_TOP')

image_transforms = [
    dict(
        type='RandomResize3D',
        scale=(800, 450),
        ratio_range=(1.0, 1.0),
        keep_ratio=True),
]

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(
        type='MultiViewWrapper',
        transforms=image_transforms,
        process_fields=('img', 'cam2img', 'lidar2cam')),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='ObjectNameFilter', classes=_base_.class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'lidar2cam', 'img_shape', 'img_path',
            'sample_idx', 'num_pts_feats', 'box_type_3d', 'box_mode_3d'
        ]),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(
        type='MultiViewWrapper',
        transforms=image_transforms,
        process_fields=('img', 'cam2img', 'lidar2cam')),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=True),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points'],
        meta_keys=[
            'cam2img', 'lidar2cam', 'img_shape', 'img_path',
            'sample_idx', 'num_pts_feats', 'box_type_3d', 'box_mode_3d'
        ]),
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=input_modality))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality))
test_dataloader = val_dataloader

model = dict(
    _delete_=True,
    type='NuScenesMBTBEVFusionDetector',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        voxel=True,
        voxel_layer=_base_.model.data_preprocessor.voxel_layer),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(2, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    voxel_encoder=_base_.model.pts_voxel_encoder,
    middle_encoder=_base_.model.pts_middle_encoder,
    backbone=_base_.model.pts_backbone,
    neck=_base_.model.pts_neck,
    bbox_head=_base_.model.pts_bbox_head,
    train_cfg=_base_.model.train_cfg.pts,
    test_cfg=_base_.model.test_cfg.pts,
    fusion_level=0,
    fusion_module=dict(
        type='NuScenesSymmetricMBTBEVFusion',
        in_img_channels=256,
        in_lidar_channels=256,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        token_grid_size=(16, 16),
        num_bottleneck_tokens=4,
        num_layers=4,
        attn_dropout=0.0,
        proj_dropout=0.1,
        point_cloud_range=_base_.point_cloud_range,
        height_anchors=(-1.5, -0.5, 0.5, 1.5),
        use_local_camera_residual=False,
        lidar_token_drop_prob=0.15,
        lidar_bev_drop_prob=0.10,
        camera_aux_num_classes=10),
    camera_aux_loss_weight=0.25,
    camera_aux_focal_alpha=0.25,
    camera_aux_focal_gamma=2.0)

optim_wrapper = dict(
    accumulative_counts=4,
    clip_grad=dict(max_norm=35, norm_type=2))
work_dir = 'work_dirs/mbt_bev_nuscenes_mini'
val_evaluator = dict(
    jsonfile_prefix='work_dirs/mbt_bev_nuscenes_mini/results')
test_evaluator = val_evaluator
