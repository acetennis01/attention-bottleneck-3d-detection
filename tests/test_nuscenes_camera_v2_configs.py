"""Resolved configuration contracts for independent larger nuScenes runs."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from mmengine.config import Config

from projects.myfusion.hooks.camera_v2_masking import CameraV2MaskingHook

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return Config.fromfile(ROOT / 'configs' / name)


def test_nuscenes_v2_preserves_dataset_geometry_head_and_evaluation():
    old = load('my_fusion_mbt_bev_nuscenes.py')
    cfg = load('nuscenes_camera_v2.py')
    assert cfg.data_root == old.data_root == 'data/nuscenes_full/'
    assert cfg.point_cloud_range == [-50, -50, -5, 50, 50, 3]
    assert cfg.train_pipeline == old.train_pipeline
    assert cfg.test_pipeline == old.test_pipeline
    assert cfg.class_names == old.class_names and len(cfg.class_names) == 10
    assert cfg.model.bbox_head == old.model.bbox_head
    assert cfg.model.neck == old.model.neck
    assert cfg.model.data_preprocessor == old.model.data_preprocessor
    assert cfg.model.type == 'NuScenesCameraFocusedMBTBEVFusionDetectorV2'
    fusion = cfg.model.fusion_module
    assert fusion.in_lidar_channels == 256
    assert fusion.embed_dim == 256 and fusion.num_bottleneck_tokens == 8
    assert fusion.camera_aux_num_classes == 10
    assert fusion.point_cloud_range == cfg.point_cloud_range
    assert 'fusion_residual_max_scale' not in fusion
    assert 'lidar_token_drop_prob' not in fusion
    assert cfg.model.img_backbone.depth == 50
    assert cfg.model.img_backbone.frozen_stages == 1
    assert cfg.model.img_neck.in_channels == [512, 1024, 2048]
    assert cfg.train_dataloader.batch_size == 1
    assert cfg.optim_wrapper.accumulative_counts == 4
    assert cfg.train_cfg.max_epochs == 24
    assert 'by_epoch' not in cfg.train_cfg
    assert cfg.custom_hooks[0].freeze_epochs == 5
    assert cfg.custom_hooks[1].type == 'CameraV2MaskingHook'
    assert cfg.randomness.seed == 0
    assert cfg.resume is False
    assert cfg.load_from == old.load_from
    assert 'pointpillars_nuscenes' in cfg.load_from
    assert cfg.default_hooks.checkpoint.save_best.endswith('/NDS')
    assert cfg.work_dir != old.work_dir


def test_nuscenes_b16_changes_only_token_count_and_output_paths():
    first = load('nuscenes_camera_v2.py').to_dict()
    second = load('nuscenes_camera_v2_b16.py').to_dict()
    assert first['model']['fusion_module']['num_bottleneck_tokens'] == 8
    assert second['model']['fusion_module']['num_bottleneck_tokens'] == 16
    assert first['work_dir'] != second['work_dir']
    for cfg in (first, second):
        del cfg['model']['fusion_module']['num_bottleneck_tokens']
        del cfg['work_dir']
        del cfg['val_evaluator']['jsonfile_prefix']
        del cfg['test_evaluator']['jsonfile_prefix']
    assert first == second


def test_nuscenes_smoke_is_bounded_and_does_not_launch_full_validation():
    cfg = load('nuscenes_camera_v2_smoke.py')
    assert cfg.train_cfg.type == 'IterBasedTrainLoop'
    assert cfg.train_cfg.max_iters == 8
    assert 'max_epochs' not in cfg.train_cfg and 'by_epoch' not in cfg.train_cfg
    assert cfg.optim_wrapper.accumulative_counts == 4
    assert cfg.val_cfg is None and cfg.val_dataloader is None
    assert cfg.val_evaluator is None and cfg.param_scheduler == []
    assert cfg.default_hooks.checkpoint.by_epoch is False


def test_no_mask_restart_changes_only_masking_and_output_paths():
    first = load('nuscenes_camera_v2.py').to_dict()
    second = load('nuscenes_camera_v2_no_mask.py').to_dict()
    assert first['model']['fusion_module']['lidar_spatial_drop_prob'] == 0.1
    assert second['model']['fusion_module']['lidar_spatial_drop_prob'] == 0.0
    assert second['custom_hooks'][1]['max_drop_prob'] == 0.0
    assert second['resume'] is False
    assert second['load_from'] == first['load_from']
    assert first['work_dir'] != second['work_dir']
    for cfg in (first, second):
        del cfg['model']['fusion_module']['lidar_spatial_drop_prob']
        del cfg['custom_hooks'][1]['max_drop_prob']
        del cfg['work_dir']
        del cfg['val_evaluator']['jsonfile_prefix']
        del cfg['test_evaluator']['jsonfile_prefix']
    assert first == second


def test_no_mask_hook_never_reenables_masking_even_on_late_resume():
    cfg = load('nuscenes_camera_v2_no_mask.py')
    kwargs = dict(cfg.custom_hooks[1])
    kwargs.pop('type')
    hook = CameraV2MaskingHook(**kwargs)
    fusion = SimpleNamespace(lidar_spatial_drop_prob=0.1)
    runner = SimpleNamespace(model=SimpleNamespace(fusion_module=fusion),
                             logger=SimpleNamespace(info=lambda text: None))
    for epoch in [0, 1, 2, 3, 4, 5, 23, 40, 2]:
        runner.epoch = epoch
        hook.before_train_epoch(runner)
        assert fusion.lidar_spatial_drop_prob == 0.0


def test_no_mask_smoke_checks_actual_zero_mask_policy():
    cfg = load('nuscenes_camera_v2_no_mask_smoke.py')
    assert cfg.model.fusion_module.lidar_spatial_drop_prob == 0.0
    assert cfg.custom_hooks[1].max_drop_prob == 0.0
    assert cfg.train_cfg.type == 'IterBasedTrainLoop'
    assert cfg.train_cfg.max_iters == 8
    assert cfg.val_cfg is None and cfg.val_evaluator is None


@pytest.mark.parametrize('mode,transform', [
    ('black', 'BlackImage'), ('shuffled', 'ShuffleNuScenesImagePaths')])
def test_no_mask_ablations_preserve_model_and_only_change_camera_input(
        mode, transform):
    original = load('nuscenes_camera_v2_no_mask.py')
    cfg = load(f'nuscenes_camera_v2_no_mask_{mode}_eval.py')
    assert cfg.model == original.model
    pipeline = [step for step in cfg.test_pipeline if step.type != transform]
    assert pipeline == original.test_pipeline
    assert cfg.test_dataloader.dataset.pipeline == cfg.test_pipeline
    assert cfg.test_dataloader == cfg.val_dataloader
