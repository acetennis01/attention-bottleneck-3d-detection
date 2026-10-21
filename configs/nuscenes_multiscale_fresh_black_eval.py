"""Black-camera evaluation of the new multi-scale architecture."""
_base_ = ['./nuscenes_multiscale_fresh.py']
custom_imports = dict(
    imports=_base_.custom_imports.imports + [
        'projects.myfusion.transforms.image_ablation'], allow_failed_imports=False)
test_pipeline = _base_.test_pipeline.copy()
test_pipeline.insert(2, dict(type='BlackImage'))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
work_dir = 'work_dirs/nuscenes_multiscale_fresh_analysis/black'
val_evaluator = dict(jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
