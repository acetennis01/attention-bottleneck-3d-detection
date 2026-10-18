"""Camera corruption evaluation of a trained V2 checkpoint, not retraining."""

_base_ = ['./kitti_camera_v2.py']

custom_imports = dict(
    imports=[
        'projects.myfusion.fusion.mbt_bev_camera_v2',
        'projects.myfusion.fusion.mbt_detector_camera_v2',
        'projects.myfusion.hooks.staged_fusion_training',
        'projects.myfusion.hooks.camera_v2_masking',
        'projects.myfusion.hooks.fusion_diagnostics',
        'projects.myfusion.transforms.image_ablation',
    ], allow_failed_imports=False)

test_pipeline = [
    dict(type='mmcv.LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1280, 384), keep_ratio=True),
    dict(type='BlackImage'),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4,
         backend_args=None),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'points']),
]
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
work_dir = 'work_dirs/kitti_camera_v2_black_eval'
val_evaluator = dict(pklfile_prefix=work_dir + '/predictions')
test_evaluator = dict(pklfile_prefix=work_dir + '/predictions')
