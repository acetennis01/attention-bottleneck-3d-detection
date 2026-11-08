"""Fresh gated/dropout training; retain audited TransFusion/Swin initializers."""
_base_ = ['./transfusion_l_mbt_swint_nuscenes.py']
custom_imports = dict(imports=_base_.custom_imports.imports + [
    'projects.myfusion.fusion.mbt_bev_gated_nuscenes'],
    allow_failed_imports=False)
model = dict(fusion_module=dict(
    type='NuScenesGatedCameraMBTBEVFusion',
    camera_drop_prob=.2, camera_view_drop_prob=.15,
    use_bottleneck_gate=True, camera_mode='present', bypass_missing=True))
work_dir = 'work_dirs/transfusion_l_gated_swint_b8_seed0'
val_evaluator = dict(jsonfile_prefix=work_dir + '/validation')
test_evaluator = val_evaluator
