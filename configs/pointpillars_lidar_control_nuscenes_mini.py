"""LiDAR-only PointPillars control on the official nuScenes mini split."""

_base_ = [
    '../../../configs/pointpillars/'
    'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
]

data_root = 'data/nuscenes/'
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
metainfo = dict(classes=class_names, version='v1.0-mini')
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

# Deliberately omit geometric augmentation so this control remains directly
# comparable with calibrated camera fusion.
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=True,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points']),
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_mini_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        metainfo=metainfo))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_mini_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'nuscenes_mini_infos_val.pkl',
    jsonfile_prefix='work_dirs/pointpillars_nuscenes_mini/results')
test_evaluator = val_evaluator

randomness = dict(seed=0, deterministic=False)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_last=True,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/NDS',
        rule='greater'))
work_dir = 'work_dirs/pointpillars_nuscenes_mini'
