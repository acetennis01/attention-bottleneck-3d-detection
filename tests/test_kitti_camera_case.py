"""Camera ablation pipeline construction must preserve model/data settings."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    '_kitti_camera_case_tests',
    Path(__file__).resolve().parents[1] / 'tools/evaluate_kitti_camera_case.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pipeline():
    return [dict(type='mmcv.LoadImageFromFile'),
            dict(type='Resize', scale=(1280, 384), keep_ratio=True),
            dict(type='LoadPointsFromFile', load_dim=4, use_dim=4),
            dict(type='PointsRangeFilter', point_cloud_range=[0, -40, -3, 70, 40, 1]),
            dict(type='Pack3DDetInputs', keys=['img', 'points'])]


@pytest.mark.parametrize('mode', ['correct', 'black', 'shuffled'])
def test_only_camera_condition_is_changed(mode):
    original = pipeline()
    changed = module.camera_pipeline(original, mode, 'val.txt', seed=0)
    assert original == pipeline()
    assert changed is not original
    assert [step for step in changed if step['type'] not in
            ('BlackImage', 'ShuffleKittiImagePath')] == original
    types = [step['type'] for step in changed]
    if mode == 'black':
        assert types.index('Resize') < types.index('BlackImage')
        assert types.index('BlackImage') < types.index('Pack3DDetInputs')
    elif mode == 'shuffled':
        assert types.index('ShuffleKittiImagePath') < types.index('mmcv.LoadImageFromFile')
        assert changed[0]['split_file'] == 'val.txt'
        assert changed[0]['seed'] == 0


def test_lidar_correct_mode_is_unchanged():
    original = [dict(type='LoadPointsFromFile')]
    assert module.camera_pipeline(original, 'correct', 'val.txt') == original
    with pytest.raises(ValueError):
        module.camera_pipeline(original, 'black', 'val.txt')


def test_rejects_double_corruption():
    original = pipeline()
    original.insert(2, dict(type='BlackImage'))
    with pytest.raises(ValueError):
        module.camera_pipeline(original, 'shuffled', 'val.txt')
