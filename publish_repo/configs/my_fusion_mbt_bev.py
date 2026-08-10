"""Three-class KITTI PointPillars with geometry-preserving symmetric MBT."""

_base_ = [
    '../../../configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
]

custom_imports = dict(
    imports=[
        'projects.myfusion.fusion.mbt_bev',
        'projects.myfusion.fusion.mbt_detector',
    ],
    allow_failed_imports=False,
)

data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# Geometry-changing point-cloud augmentations are intentionally disabled here:
# they would desynchronize the camera image and LiDAR/boxes. Photometric image
# augmentation can be added later without changing calibration.
train_pipeline = [
    dict(type='mmcv.LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1280, 384), keep_ratio=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(type='mmcv.LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1280, 384), keep_ratio=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'points']),
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        dataset=dict(
            data_prefix=dict(
                pts='training/velodyne_reduced',
                img='training/image_2'),
            pipeline=train_pipeline,
            modality=input_modality,
            metainfo=metainfo)))
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo))
test_dataloader = val_dataloader

model = dict(
    type='MBTBEVFusionDetector',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(2, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet18')),
    fusion_module=dict(
        type='SymmetricMBTBEVFusion',
        in_img_channels=256,
        in_lidar_channels=384,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        token_grid_size=(16, 16),
        num_bottleneck_tokens=4,
        num_layers=4,
        attn_dropout=0.0,
        proj_dropout=0.1,
        bottleneck_init_std=0.02),
)

# The camera backbone adds memory relative to PointPillars.
optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))

# Keep recoverable progress during long runs without accumulating checkpoints.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=3, save_last=True))

# Benchmark claims must use the upstream KITTI metric logic.  A CPU rotated-IoU
# backend is acceptable only when it is regression-tested to match the official
# criterion semantics.
work_dir = 'work_dirs/my_fusion_mbt_bev'
