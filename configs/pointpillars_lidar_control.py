"""Controlled LiDAR-only PointPillars baseline for the KITTI MBT study."""

_base_ = [
    '../../../configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
]

data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

# Match the LiDAR half of the fusion pipeline. Geometry-changing augmentation
# is omitted because it would desynchronize camera and LiDAR inputs.
train_pipeline = [
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
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points']),
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        dataset=dict(
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=train_pipeline,
            modality=input_modality,
            metainfo=metainfo)))
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo))
test_dataloader = val_dataloader

randomness = dict(seed=0, deterministic=False)
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=2)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_last=True,
        save_best=(
            'Kitti metric/pred_instances_3d/KITTI/'
            'Overall_3D_AP40_moderate'),
        rule='greater'))

work_dir = 'work_dirs/pointpillars_lidar_control'
val_evaluator = dict(pklfile_prefix=work_dir)
test_evaluator = dict(pklfile_prefix=work_dir)
