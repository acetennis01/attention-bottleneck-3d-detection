"""Resolve the training contracts in a complete MMDetection3D checkout."""
from pathlib import Path
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return Config.fromfile(ROOT / 'configs' / name)


def test_kitti_arms_preserve_frozen_baseline_and_unique_outputs():
    names = ['kitti_gated_b8.py', 'kitti_gated_average.py',
             'kitti_gated_dropout_only.py', 'kitti_gated_gate_only.py',
             'kitti_gated_absent_control.py']
    configs = [load(name) for name in names]
    assert len({cfg.work_dir for cfg in configs}) == len(names)
    for cfg in configs:
        assert cfg.custom_hooks[0].freeze_epochs >= cfg.train_cfg.max_epochs
        assert all(h.type != 'CameraV2MaskingHook' for h in cfg.custom_hooks)
        assert cfg.model.fusion_module.lidar_spatial_drop_prob == 0
        assert cfg.optim_wrapper.accumulative_counts == 8
        assert cfg.train_dataloader.batch_size == 6
        assert cfg.load_from == configs[0].load_from
        assert cfg.model.img_backbone.depth == 50


def test_gate_and_dropout_form_two_by_two_control():
    arms = [('kitti_gated_b8.py', True, .2),
            ('kitti_gated_average.py', False, 0),
            ('kitti_gated_dropout_only.py', False, .2),
            ('kitti_gated_gate_only.py', True, 0)]
    for name, gate, dropout in arms:
        fusion = load(name).model.fusion_module
        assert fusion.use_bottleneck_gate is gate
        assert fusion.camera_drop_prob == dropout
        assert fusion.camera_mode == 'present'
        assert fusion.bypass_missing


def test_absent_control_trains_residual_instead_of_trivial_bypass():
    fusion = load('kitti_gated_absent_control.py').model.fusion_module
    assert fusion.camera_mode == 'missing'
    assert fusion.bypass_missing is False


def test_transfusion_retains_audited_initializers_and_full_training_budget():
    old = load('transfusion_l_mbt_swint_nuscenes.py')
    cfg = load('transfusion_l_gated_swint_nuscenes.py')
    assert cfg.model.freeze_lidar
    assert cfg.train_cfg == old.train_cfg
    assert cfg.train_cfg.max_iters == 160000
    assert cfg.optim_wrapper == old.optim_wrapper
    assert cfg.custom_hooks == old.custom_hooks
    assert cfg.model.img_backbone == old.model.img_backbone
    assert cfg.camera_checkpoint_sha256 == old.camera_checkpoint_sha256
    assert cfg.model.fusion_module.camera_view_drop_prob == .15
    assert cfg.model.fusion_module.camera_drop_prob == .2
    assert cfg.work_dir != old.work_dir
