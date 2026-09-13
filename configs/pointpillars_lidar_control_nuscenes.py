"""LiDAR-only PointPillars control on nuScenes v1.0 trainval."""

_base_ = ['./pointpillars_lidar_control_nuscenes_mini.py']

data_root = 'data/nuscenes_full/'
metainfo = dict(classes=_base_.class_names, version='v1.0-trainval')

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        metainfo=metainfo))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    jsonfile_prefix='work_dirs/pointpillars_nuscenes/results')
test_evaluator = val_evaluator

work_dir = 'work_dirs/pointpillars_nuscenes'
