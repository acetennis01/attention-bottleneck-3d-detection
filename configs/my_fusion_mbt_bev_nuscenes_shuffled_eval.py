"""Evaluate nuScenes MBT with deterministic mismatched six-camera images."""

_base_ = ['./my_fusion_mbt_bev_nuscenes.py']

custom_imports = dict(
    imports=[
        'projects.myfusion.fusion.mbt_bev',
        'projects.myfusion.fusion.mbt_detector',
        'projects.myfusion.fusion.mbt_bev_nuscenes',
        'projects.myfusion.fusion.mbt_detector_nuscenes',
        'projects.myfusion.hooks.fusion_diagnostics',
        'projects.myfusion.hooks.staged_fusion_training',
        'projects.myfusion.transforms.image_ablation',
    ],
    allow_failed_imports=False)

test_pipeline = _base_.test_pipeline.copy()
test_pipeline.insert(
    0,
    dict(
        type='ShuffleNuScenesImagePaths',
        info_file='data/nuscenes_full/nuscenes_infos_val.pkl',
        seed=0))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

work_dir = 'work_dirs/nuscenes_analysis/shuffled'
val_evaluator = dict(
    jsonfile_prefix='work_dirs/nuscenes_analysis/shuffled/results')
test_evaluator = val_evaluator
