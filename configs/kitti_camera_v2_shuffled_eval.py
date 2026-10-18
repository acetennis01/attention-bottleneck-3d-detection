"""Break camera/scene correspondence at evaluation of a trained V2 model."""

_base_ = ['./kitti_camera_v2_black_eval.py']

test_pipeline = [
    dict(type='ShuffleKittiImagePath',
         split_file='data/kitti/ImageSets/val.txt', seed=0),
    dict(type='mmcv.LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1280, 384), keep_ratio=True),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4,
         backend_args=None),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'points']),
]
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
work_dir = 'work_dirs/kitti_camera_v2_shuffled_eval'
val_evaluator = dict(pklfile_prefix=work_dir + '/predictions')
test_evaluator = dict(pklfile_prefix=work_dir + '/predictions')
