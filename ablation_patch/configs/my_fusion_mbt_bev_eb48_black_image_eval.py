"""Evaluate the best MBT checkpoint with all-black camera images."""

_base_ = ['./my_fusion_mbt_bev_eb48.py']

custom_imports = dict(
    imports=[
        'projects.myfusion.fusion.mbt_bev',
        'projects.myfusion.fusion.mbt_detector',
        'projects.myfusion.hooks.fusion_diagnostics',
        'projects.myfusion.transforms.image_ablation',
    ],
    allow_failed_imports=False,
)

backend_args = None
test_pipeline = [
    dict(type='mmcv.LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1280, 384), keep_ratio=True),
    dict(type='BlackImage'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'points']),
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

work_dir = 'work_dirs/mbt_bev_best_black_image_eval'
val_evaluator = dict(pklfile_prefix=work_dir)
test_evaluator = dict(pklfile_prefix=work_dir)
