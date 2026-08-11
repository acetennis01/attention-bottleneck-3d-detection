"""Ensure the LiDAR control and MBT experiment controls stay matched."""

from pathlib import Path

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
