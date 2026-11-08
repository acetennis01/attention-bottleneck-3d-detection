"""Run 4 must change the fusion graph and output paths only."""
from pathlib import Path
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]


def test_full_attention_preserves_part_b_recipe():
    mbt = Config.fromfile(ROOT / 'configs/kitti_gated_b8.py').to_dict()
    full = Config.fromfile(ROOT / 'configs/kitti_full_attention.py').to_dict()
    assert full['model']['fusion_module']['type'] == 'FullAttentionBEVFusion'
    assert 'num_bottleneck_tokens' not in full['model']['fusion_module']
    assert 'use_bottleneck_gate' not in full['model']['fusion_module']
    # Remove config construction helper, which is not a runner argument.
    full.pop('full_attention', None)
    full['custom_imports']['imports'].remove(
        'projects.myfusion.fusion.full_attention_bev')
    for key in ('num_bottleneck_tokens', 'use_bottleneck_gate'):
        mbt['model']['fusion_module'].pop(key)
    for cfg in (full, mbt):
        cfg['model']['fusion_module'].pop('type')
        cfg.pop('work_dir')
        cfg['val_evaluator'].pop('pklfile_prefix')
        cfg['test_evaluator'].pop('pklfile_prefix')
    assert full == mbt
