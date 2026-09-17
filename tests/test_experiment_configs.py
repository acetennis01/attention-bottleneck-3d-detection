"""Configuration checks for the final KITTI experiments."""

from pathlib import Path

import pytest
from mmengine.config import Config


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs'


def _load(name: str) -> Config:
    return Config.fromfile(CONFIG_DIR / name)


def test_effective_batch_48_baseline_matches_fusion_recipe():
    lidar = _load('pointpillars_lidar_control_eb48.py')
    fusion = _load('my_fusion_mbt_bev_eb48.py')

    for cfg in (lidar, fusion):
        assert cfg.train_dataloader.batch_size == 6
        assert cfg.optim_wrapper.accumulative_counts == 8
        assert cfg.train_dataloader.batch_size * 8 == 48
        assert cfg.randomness.seed == 0
        assert cfg.train_cfg.max_epochs == 80
        assert cfg.train_cfg.val_interval == 2
        assert cfg.param_scheduler[2].eta_min == pytest.approx(0.85)
        assert cfg.param_scheduler[3].eta_min == pytest.approx(0.95)

    assert lidar.class_names == fusion.class_names
    assert lidar.point_cloud_range == fusion.point_cloud_range


def test_aligned_config_uses_kitti_geometry():
    cfg = _load('my_fusion_mbt_bev_aligned_eb48.py')
    fusion = cfg.model.fusion_module

    assert fusion.point_cloud_range == cfg.point_cloud_range
    assert tuple(fusion.height_anchors) == (-1.5, -0.75, 0.0, 0.75)


def test_camera_focused_config_matches_reported_model():
    cfg = _load('my_fusion_mbt_bev_camera_focused_eb48.py')
    fusion = cfg.model.fusion_module

    assert fusion.type == 'SymmetricMBTBEVFusion'
    assert fusion.num_bottleneck_tokens == 4
    assert fusion.num_layers == 4
    assert fusion.num_heads == 8
    assert fusion.use_local_camera_residual is False
    assert fusion.lidar_token_drop_prob == pytest.approx(0.15)
    assert fusion.lidar_bev_drop_prob == pytest.approx(0.10)
    assert fusion.camera_aux_num_classes == 3
    assert cfg.model.camera_aux_loss_weight == pytest.approx(0.25)
    assert cfg.train_cfg.max_epochs == 40
    assert cfg.train_cfg.val_interval == 2
    assert cfg.custom_hooks[0].type == 'StagedFusionTrainingHook'
    assert cfg.custom_hooks[0].freeze_epochs == 5
    assert 'pointpillars_lidar_control' in cfg.load_from


def test_camera_focused_smoke_is_bounded():
    cfg = _load('my_fusion_mbt_bev_camera_focused_eb48_smoke.py')

    assert cfg.train_cfg.type == 'IterBasedTrainLoop'
    assert cfg.train_cfg.max_iters == 16
    assert cfg.val_cfg is None
    assert cfg.val_dataloader is None
    assert cfg.val_evaluator is None
    assert cfg.param_scheduler == []


def test_nuscenes_camera_focused_config_matches_kitti_methodology():
    cfg = _load('my_fusion_mbt_bev_nuscenes.py')

    assert cfg.train_cfg.type == 'EpochBasedTrainLoop'
    assert cfg.train_cfg.max_epochs == 24
    assert cfg.train_cfg.val_begin == 2
    assert cfg.optim_wrapper.accumulative_counts == 4
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(3e-4)
    assert cfg.custom_hooks[0].type == 'StagedFusionTrainingHook'
    assert cfg.custom_hooks[0].freeze_epochs == 5
    assert cfg.custom_hooks[1].type == 'FusionDiagnosticsHook'
    assert 'pointpillars_nuscenes' in cfg.load_from

    custom_keys = cfg.optim_wrapper.paramwise_cfg.custom_keys
    assert custom_keys.img_backbone.lr_mult == pytest.approx(2.0)
    assert custom_keys.fusion_module.lr_mult == pytest.approx(2.0)
    assert custom_keys.voxel_encoder.lr_mult == pytest.approx(0.5)
    assert custom_keys.bbox_head.lr_mult == pytest.approx(0.5)


@pytest.mark.parametrize(
    'name, transform',
    [
        ('my_fusion_mbt_bev_camera_focused_black_eval.py', 'BlackImage'),
        (
            'my_fusion_mbt_bev_camera_focused_shuffled_eval.py',
            'ShuffleKittiImagePath',
        ),
    ],
)
def test_camera_ablation_configs(name: str, transform: str):
    cfg = _load(name)
    assert transform in [step.type for step in cfg.test_pipeline]
    assert cfg.test_dataloader == cfg.val_dataloader
