"""Run 4 control for the modified frozen R50 KITTI Part B recipe."""
from copy import deepcopy

_base_ = ['./kitti_gated_b8.py']
custom_imports = dict(imports=_base_.custom_imports.imports + [
    'projects.myfusion.fusion.full_attention_bev'], allow_failed_imports=False)

full_attention = deepcopy(_base_.model.fusion_module)
full_attention.pop('num_bottleneck_tokens')
full_attention.pop('use_bottleneck_gate')
full_attention.update(type='FullAttentionBEVFusion', _delete_=True)
model = dict(fusion_module=full_attention)
work_dir = 'work_dirs/kitti_full_attention_r50_seed0'
val_evaluator = dict(pklfile_prefix=work_dir + '/val_predictions')
test_evaluator = dict(pklfile_prefix=work_dir + '/test_predictions')
