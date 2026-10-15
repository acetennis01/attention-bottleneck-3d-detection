auto_scale_lr = dict(base_batch_size=48, enable=False)
backend_args = None
class_names = [
    'Pedestrian',
    'Cyclist',
    'Car',
]
custom_hooks = [
    dict(freeze_epochs=5, type='StagedFusionTrainingHook'),
    dict(interval=8, type='FusionDiagnosticsHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.myfusion.fusion.mbt_bev_kitti_v1',
        'projects.myfusion.fusion.mbt_detector',
        'projects.myfusion.hooks.staged_fusion_training',
        'projects.myfusion.hooks.fusion_diagnostics',
    ])
data_root = 'data/kitti/'
dataset_type = 'KittiDataset'
db_sampler = dict(
    backend_args=None,
    classes=[
        'Pedestrian',
        'Cyclist',
        'Car',
    ],
    data_root='data/kitti/',
    info_path='data/kitti/kitti_dbinfos_train.pkl',
    points_loader=dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    prepare=dict(
        filter_by_difficulty=[
            -1,
        ],
        filter_by_min_points=dict(Car=5, Cyclist=5, Pedestrian=5)),
    rate=1.0,
    sample_groups=dict(Car=15, Cyclist=15, Pedestrian=15))
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        save_best=None,
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=8, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
epoch_num = 40
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=True, use_lidar=True)
launcher = 'none'
load_from = 'work_dirs/pointpillars_lidar_control_eb48_stable_momentum_seed0/best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_epoch_28.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.001
metainfo = dict(classes=[
    'Pedestrian',
    'Cyclist',
    'Car',
])
model = dict(
    backbone=dict(
        in_channels=64,
        layer_nums=[
            3,
            5,
            5,
        ],
        layer_strides=[
            2,
            2,
            2,
        ],
        out_channels=[
            64,
            128,
            256,
        ],
        type='SECOND'),
    bbox_head=dict(
        anchor_generator=dict(
            ranges=[
                [
                    0,
                    -39.68,
                    -0.6,
                    69.12,
                    39.68,
                    -0.6,
                ],
                [
                    0,
                    -39.68,
                    -0.6,
                    69.12,
                    39.68,
                    -0.6,
                ],
                [
                    0,
                    -39.68,
                    -1.78,
                    69.12,
                    39.68,
                    -1.78,
                ],
            ],
            reshape_out=False,
            rotations=[
                0,
                1.57,
            ],
            sizes=[
                [
                    0.8,
                    0.6,
                    1.73,
                ],
                [
                    1.76,
                    0.6,
                    1.73,
                ],
                [
                    3.9,
                    1.6,
                    1.56,
                ],
            ],
            type='AlignedAnchor3DRangeGenerator'),
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        diff_rad_by_sin=True,
        feat_channels=384,
        in_channels=384,
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=2.0,
            type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_dir=dict(
            loss_weight=0.2, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        num_classes=3,
        type='Anchor3DHead',
        use_direction_classifier=True),
    camera_aux_focal_alpha=0.25,
    camera_aux_focal_gamma=2.0,
    camera_aux_loss_weight=0.25,
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            max_voxels=(
                16000,
                40000,
            ),
            point_cloud_range=[
                0,
                -39.68,
                -3,
                69.12,
                39.68,
                1,
            ],
            voxel_size=[
                0.16,
                0.16,
                4,
            ])),
    fusion_module=dict(
        attn_dropout=0.0,
        bottleneck_init_std=0.02,
        camera_aux_num_classes=3,
        embed_dim=128,
        height_anchors=(
            -1.5,
            -0.75,
            0.0,
            0.75,
        ),
        in_img_channels=256,
        in_lidar_channels=384,
        lidar_bev_drop_prob=0.1,
        lidar_token_drop_prob=0.15,
        mlp_ratio=4.0,
        num_bottleneck_tokens=16,
        num_heads=8,
        num_layers=4,
        point_cloud_range=[
            0,
            -39.68,
            -3,
            69.12,
            39.68,
            1,
        ],
        proj_dropout=0.1,
        token_grid_size=(
            16,
            16,
        ),
        type='KittiV1SymmetricMBTBEVFusion',
        use_local_camera_residual=False),
    img_backbone=dict(
        depth=18,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet18', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(2, ),
        style='pytorch',
        type='mmdet.ResNet'),
    middle_encoder=dict(
        in_channels=64, output_shape=[
            496,
            432,
        ], type='PointPillarsScatter'),
    neck=dict(
        in_channels=[
            64,
            128,
            256,
        ],
        out_channels=[
            128,
            128,
            128,
        ],
        type='SECONDFPN',
        upsample_strides=[
            1,
            2,
            4,
        ]),
    test_cfg=dict(
        max_num=50,
        min_bbox_size=0,
        nms_across_levels=False,
        nms_pre=100,
        nms_thr=0.01,
        score_thr=0.1,
        use_rotate_nms=True),
    train_cfg=dict(
        allowed_border=0,
        assigner=[
            dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                min_pos_iou=0.35,
                neg_iou_thr=0.35,
                pos_iou_thr=0.5,
                type='Max3DIoUAssigner'),
            dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                min_pos_iou=0.35,
                neg_iou_thr=0.35,
                pos_iou_thr=0.5,
                type='Max3DIoUAssigner'),
            dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                min_pos_iou=0.45,
                neg_iou_thr=0.45,
                pos_iou_thr=0.6,
                type='Max3DIoUAssigner'),
        ],
        debug=False,
        pos_weight=-1),
    type='MBTBEVFusionDetector',
    voxel_encoder=dict(
        feat_channels=[
            64,
        ],
        in_channels=4,
        point_cloud_range=[
            0,
            -39.68,
            -3,
            69.12,
            39.68,
            1,
        ],
        type='PillarFeatureNet',
        voxel_size=[
            0.16,
            0.16,
            4,
        ],
        with_distance=False))
optim_wrapper = dict(
    accumulative_counts=8,
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        betas=(
            0.95,
            0.99,
        ), lr=0.0003, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.5),
            bbox_head=dict(lr_mult=0.5),
            fusion_module=dict(lr_mult=2.0),
            img_backbone=dict(lr_mult=2.0),
            middle_encoder=dict(lr_mult=0.5),
            neck=dict(lr_mult=0.5),
            voxel_encoder=dict(lr_mult=0.5))),
    type='OptimWrapper')
param_scheduler = []
point_cloud_range = [
    0,
    -39.68,
    -3,
    69.12,
    39.68,
    1,
]
randomness = dict(deterministic=False, seed=0)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='training/image_2', pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        indices=32,
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=True),
        pipeline=[
            dict(backend_args=None, type='mmcv.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1280,
                384,
            ), type='Resize'),
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                point_cloud_range=[
                    0,
                    -39.68,
                    -3,
                    69.12,
                    39.68,
                    1,
                ],
                type='PointsRangeFilter'),
            dict(keys=[
                'img',
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/home/yab/mmdetection3d/work_dirs/kitti_bottleneck_smoke_kpJcL2/b16/val_infos_smoke.pkl',
    backend_args=None,
    metric='bbox',
    pklfile_prefix=
    '/home/yab/mmdetection3d/work_dirs/kitti_bottleneck_smoke_kpJcL2/b16/test_predictions',
    type='KittiMetric')
test_pipeline = [
    dict(backend_args=None, type='mmcv.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1280,
        384,
    ), type='Resize'),
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(
        point_cloud_range=[
            0,
            -39.68,
            -3,
            69.12,
            39.68,
            1,
        ],
        type='PointsRangeFilter'),
    dict(keys=[
        'img',
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        ann_file='kitti_infos_train.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='training/image_2', pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        indices=96,
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=True),
        pipeline=[
            dict(backend_args=None, type='mmcv.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1280,
                384,
            ), type='Resize'),
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                point_cloud_range=[
                    0,
                    -39.68,
                    -3,
                    69.12,
                    39.68,
                    1,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    0,
                    -39.68,
                    -3,
                    69.12,
                    39.68,
                    1,
                ],
                type='ObjectRangeFilter'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'img',
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='KittiDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='mmcv.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1280,
        384,
    ), type='Resize'),
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        point_cloud_range=[
            0,
            -39.68,
            -3,
            69.12,
            39.68,
            1,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            0,
            -39.68,
            -3,
            69.12,
            39.68,
            1,
        ],
        type='ObjectRangeFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'img',
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='training/image_2', pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        indices=32,
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=True),
        pipeline=[
            dict(backend_args=None, type='mmcv.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1280,
                384,
            ), type='Resize'),
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                point_cloud_range=[
                    0,
                    -39.68,
                    -3,
                    69.12,
                    39.68,
                    1,
                ],
                type='PointsRangeFilter'),
            dict(keys=[
                'img',
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/home/yab/mmdetection3d/work_dirs/kitti_bottleneck_smoke_kpJcL2/b16/val_infos_smoke.pkl',
    backend_args=None,
    metric='bbox',
    pklfile_prefix=
    '/home/yab/mmdetection3d/work_dirs/kitti_bottleneck_smoke_kpJcL2/b16/val_predictions',
    type='KittiMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.16,
    0.16,
    4,
]
work_dir = '/home/yab/mmdetection3d/work_dirs/kitti_bottleneck_smoke_kpJcL2/b16'
