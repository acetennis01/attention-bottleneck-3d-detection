"""Resolve V2 inheritance inside a normal MMDetection3D checkout."""

from pathlib import Path

import pytest
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return Config.fromfile(ROOT / 'configs' / name)


def test_v2_config_capacity_and_training_contract():
    cfg = load('kitti_camera_v2.py')
    assert cfg.model.type == 'CameraFocusedMBTBEVFusionDetectorV2'
    assert cfg.model.img_backbone.depth == 50
    assert tuple(cfg.model.img_backbone.out_indices) == (1, 2, 3)
    assert cfg.model.img_backbone.frozen_stages == 1
    assert cfg.model.img_backbone.norm_eval
    assert not cfg.model.img_backbone.norm_cfg.requires_grad
    assert cfg.model.img_neck.in_channels == [512, 1024, 2048]
    assert cfg.model.img_feature_level == 0
    fusion = cfg.model.fusion_module
    assert fusion.type == 'CameraFocusedMBTBEVFusionV2'
    assert fusion.embed_dim == 256
    assert fusion.num_bottleneck_tokens == 8
    assert fusion.num_heads == 8 and fusion.num_layers == 4
    assert fusion.use_local_camera_residual is False
    assert fusion.camera_aggregation_mode == 'valid_only'
    assert fusion.camera_aux_num_classes == 3
    assert fusion.point_cloud_range == cfg.point_cloud_range
    assert 'fusion_residual_max_scale' not in fusion
    assert 'lidar_bev_drop_prob' not in fusion
    assert cfg.train_dataloader.batch_size == 6
    assert cfg.optim_wrapper.accumulative_counts == 8
    assert cfg.randomness.seed == 0
    assert cfg.train_cfg.max_epochs == 40
    assert cfg.custom_hooks[0].freeze_epochs == 5
    assert cfg.custom_hooks[1].type == 'CameraV2MaskingHook'
    assert cfg.model.camera_aux_loss_weight == pytest.approx(0.25)
    assert 'pointpillars_lidar_control' in cfg.load_from
    assert cfg.resume is False
    assert 'Overall_3D_AP40_moderate' in cfg.default_hooks.checkpoint.save_best
    keys = cfg.optim_wrapper.paramwise_cfg.custom_keys
    assert keys.img_backbone.lr_mult == pytest.approx(0.2)
    assert keys.img_neck.lr_mult == pytest.approx(1.0)
    assert keys.backbone.lr_mult == pytest.approx(0.5)


def test_v2_smoke_has_two_updates_and_no_validation():
    cfg = load('kitti_camera_v2_smoke.py')
    assert cfg.train_cfg.type == 'IterBasedTrainLoop'
    assert 'by_epoch' not in cfg.train_cfg
    assert 'max_epochs' not in cfg.train_cfg
    assert cfg.train_cfg.max_iters == 16
    assert cfg.optim_wrapper.accumulative_counts == 8
    assert cfg.val_cfg is None and cfg.val_dataloader is None
    assert cfg.val_evaluator is None and cfg.param_scheduler == []
    assert all(hook.type != 'CameraV2MaskingHook' for hook in cfg.custom_hooks)


@pytest.mark.parametrize('name, expected', [
    ('kitti_camera_v2_black_eval.py', 'BlackImage'),
    ('kitti_camera_v2_shuffled_eval.py', 'ShuffleKittiImagePath'),
])
def test_v2_camera_evaluation_uses_same_model(name, expected):
    cfg = load(name)
    types = [step.type for step in cfg.test_pipeline]
    assert expected in types
    if expected == 'ShuffleKittiImagePath':
        assert 'BlackImage' not in types
    assert cfg.test_dataloader == cfg.val_dataloader
    assert cfg.model.fusion_module.embed_dim == 256
    assert cfg.model.type == 'CameraFocusedMBTBEVFusionDetectorV2'
    assert cfg.test_dataloader.dataset.pipeline == cfg.test_pipeline


def test_v2_no_mask_is_a_separate_training_control():
    cfg = load('kitti_camera_v2_no_mask.py')
    assert cfg.model.fusion_module.lidar_spatial_drop_prob == 0
    assert cfg.custom_hooks[1].max_drop_prob == 0
    assert cfg.work_dir != load('kitti_camera_v2.py').work_dir


def test_v2_sixteen_tokens_changes_only_token_count_and_output_paths():
    reference = load('kitti_camera_v2.py').to_dict()
    variant = load('kitti_camera_v2_b16.py').to_dict()
    assert reference['model']['fusion_module']['num_bottleneck_tokens'] == 8
    assert variant['model']['fusion_module']['num_bottleneck_tokens'] == 16
    assert reference['work_dir'] != variant['work_dir']
    for cfg in (reference, variant):
        del cfg['model']['fusion_module']['num_bottleneck_tokens']
        del cfg['work_dir']
        del cfg['val_evaluator']['pklfile_prefix']
        del cfg['test_evaluator']['pklfile_prefix']
    assert variant == reference
