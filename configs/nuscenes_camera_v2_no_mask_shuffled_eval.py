"""Mismatched-camera evaluation with fixed seed and original calibration."""

_base_ = ['./nuscenes_camera_v2_no_mask.py']

custom_imports = dict(
    imports=_base_.custom_imports.imports + [
        'projects.myfusion.transforms.image_ablation'],
    allow_failed_imports=False)
test_pipeline = _base_.test_pipeline.copy()
test_pipeline.insert(0, dict(
    type='ShuffleNuScenesImagePaths',
    info_file='data/nuscenes_full/nuscenes_infos_val.pkl', seed=0))
val_dataloader = dict(batch_size=1, num_workers=1, persistent_workers=False,
                      dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
work_dir = 'work_dirs/nuscenes_camera_v2_nomask_analysis/shuffled'
val_evaluator = dict(jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
