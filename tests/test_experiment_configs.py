"""Ensure the LiDAR control and MBT experiment controls stay matched."""

from pathlib import Path

import pytest
from mmengine.config import Config


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs'
BEST_METRIC = (
    'Kitti metric/pred_instances_3d/KITTI/'
    'Overall_3D_AP40_moderate')


def _load(name):
    return Config.fromfile(CONFIG_DIR / name)


def test_full_experiment_controls_match():
    lidar = _load('pointpillars_lidar_control.py')
    mbt = _load('my_fusion_mbt_bev.py')

    assert lidar.randomness == mbt.randomness
    assert lidar.randomness.seed == 0
    assert lidar.train_dataloader.batch_size == 1
    assert mbt.train_dataloader.batch_size == 1
    assert lidar.train_cfg == mbt.train_cfg
    assert lidar.train_cfg.max_epochs == 80
    assert lidar.train_cfg.val_interval == 2
    assert lidar.class_names == mbt.class_names
    assert lidar.point_cloud_range == mbt.point_cloud_range

    for cfg in (lidar, mbt):
        checkpoint = cfg.default_hooks.checkpoint
        assert checkpoint.interval == 1
        assert checkpoint.max_keep_ckpts == 3
        assert checkpoint.save_last is True
        assert checkpoint.save_best == BEST_METRIC
        assert checkpoint.rule == 'greater'
        assert cfg.val_evaluator.pklfile_prefix == cfg.work_dir
        assert cfg.test_evaluator.pklfile_prefix == cfg.work_dir


def test_lidar_pipeline_matches_non_image_half_of_mbt_pipeline():
    lidar = _load('pointpillars_lidar_control.py')
    mbt = _load('my_fusion_mbt_bev.py')
    image_only = {'mmcv.LoadImageFromFile', 'Resize'}

    mbt_train_types = [
        step.type for step in mbt.train_pipeline
        if step.type not in image_only
    ]
    mbt_test_types = [
        step.type for step in mbt.test_pipeline
        if step.type not in image_only
    ]
    assert [step.type for step in lidar.train_pipeline] == mbt_train_types
    assert [step.type for step in lidar.test_pipeline] == mbt_test_types
    assert lidar.input_modality == dict(use_lidar=True, use_camera=False)
    assert mbt.input_modality == dict(use_lidar=True, use_camera=True)


def test_smoke_configs_are_bounded_and_persistent():
    for name in (
            'pointpillars_lidar_control_smoke.py',
            'my_fusion_mbt_bev_smoke.py'):
        cfg = _load(name)
        assert cfg.train_cfg.max_epochs == 2
        assert cfg.train_cfg.val_interval == 1
        assert cfg.default_hooks.checkpoint.interval == 1
        assert cfg.default_hooks.checkpoint.max_keep_ckpts == 2
        assert cfg.val_evaluator.pklfile_prefix == cfg.work_dir
        assert cfg.test_evaluator.pklfile_prefix == cfg.work_dir


def test_lower_lr_experiment_changes_only_schedule_and_diagnostics():
    original = _load('my_fusion_mbt_bev.py')
    lower_lr = _load('my_fusion_mbt_bev_lr3e4.py')

    assert lower_lr.model == original.model
    assert lower_lr.train_dataloader == original.train_dataloader
    assert lower_lr.val_dataloader == original.val_dataloader
    assert lower_lr.train_cfg == original.train_cfg
    assert lower_lr.randomness == original.randomness
    assert lower_lr.optim_wrapper.optimizer.lr == 3e-4
    assert lower_lr.param_scheduler[0].eta_min == pytest.approx(3e-3)
    assert lower_lr.param_scheduler[1].eta_min == pytest.approx(3e-8)
    assert lower_lr.custom_hooks == [
        dict(type='FusionDiagnosticsHook', interval=500)]
    assert lower_lr.work_dir == 'work_dirs/mbt_bev_lr3e4_seed0'
    assert lower_lr.val_evaluator.pklfile_prefix == lower_lr.work_dir
    assert lower_lr.test_evaluator.pklfile_prefix == lower_lr.work_dir


def test_effective_batch_48_experiments_are_matched():
    lidar = _load('pointpillars_lidar_control_eb48.py')
    mbt = _load('my_fusion_mbt_bev_eb48.py')

    for cfg in (lidar, mbt):
        assert cfg.train_dataloader.batch_size == 6
        assert cfg.optim_wrapper.accumulative_counts == 8
        assert (cfg.train_dataloader.batch_size
                * cfg.optim_wrapper.accumulative_counts) == 48
        assert cfg.auto_scale_lr.base_batch_size == 48
        assert cfg.randomness.seed == 0
        assert cfg.train_cfg.max_epochs == 80
        assert cfg.train_cfg.val_interval == 2
        assert cfg.val_evaluator.pklfile_prefix == cfg.work_dir
        assert cfg.test_evaluator.pklfile_prefix == cfg.work_dir

    assert lidar.optim_wrapper.optimizer == mbt.optim_wrapper.optimizer
    assert lidar.param_scheduler == mbt.param_scheduler
    assert lidar.param_scheduler[2].eta_min == 0.85
    assert lidar.param_scheduler[3].eta_min == 0.95
    assert lidar.work_dir.endswith('eb48_stable_momentum_seed0')
    assert mbt.work_dir.endswith('eb48_stable_momentum_seed0')
    assert lidar.class_names == mbt.class_names
    assert lidar.point_cloud_range == mbt.point_cloud_range
    assert mbt.custom_hooks == [
        dict(type='FusionDiagnosticsHook', interval=100)]


def test_effective_batch_smoke_performs_two_accumulated_updates():
    cfg = _load('my_fusion_mbt_bev_eb48_smoke.py')
    assert cfg.train_dataloader.batch_size == 6
    assert cfg.optim_wrapper.accumulative_counts == 8
    assert cfg.train_cfg.type == 'IterBasedTrainLoop'
    assert cfg.train_cfg.max_iters == 16
    assert cfg.val_cfg is None
    assert cfg.val_dataloader is None
    assert cfg.val_evaluator is None
    assert cfg.param_scheduler == []
    assert cfg.custom_hooks == [
        dict(type='FusionDiagnosticsHook', interval=8)]
