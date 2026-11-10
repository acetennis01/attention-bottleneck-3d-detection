"""Camera corruption must not change the detector or LiDAR pipeline."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec = importlib.util.spec_from_file_location('_nusc_camera_case',
    Path(__file__).resolve().parents[1] / 'tools/evaluate_nuscenes_camera_case.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pipeline():
    return [dict(type='LoadMultiViewImageFromFiles', num_views=6),
            dict(type='MultiViewWrapper', transforms=[dict(type='RandomResize3D')]),
            dict(type='LoadPointsFromFile', load_dim=5),
            dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, test_mode=True),
            dict(type='Pack3DDetInputs', keys=['img', 'points'])]


@pytest.mark.parametrize('mode', ['correct', 'black', 'shuffled', 'missing'])
def test_preserves_every_non_camera_step(mode):
    source = pipeline()
    actual = module.camera_pipeline(source, mode, 'infos.pkl', seed=7)
    assert source == pipeline()
    assert actual is not source
    assert [step for step in actual if step['type'] not in module.CORRUPTIONS] == source
    types = [step['type'] for step in actual]
    if mode == 'black':
        assert types.index('MultiViewWrapper') < types.index('BlackImage')
        assert types.index('BlackImage') < types.index('LoadPointsFromFile')
    elif mode == 'shuffled':
        assert types[0] == 'ShuffleNuScenesImagePaths'
        assert actual[0]['seed'] == 7
        assert actual[0]['info_file'] == 'infos.pkl'


def test_rejects_double_corruption():
    source = pipeline() + [dict(type='BlackImage')]
    with pytest.raises(ValueError):
        module.camera_pipeline(source, 'correct', 'infos.pkl')


@pytest.mark.parametrize('mode', ['correct', 'black', 'shuffled', 'missing'])
def test_requires_camera_loader(mode):
    with pytest.raises(ValueError):
        module.camera_pipeline([dict(type='LoadPointsFromFile')], mode, 'infos.pkl')


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        module.camera_pipeline(pipeline(), 'unknown', 'infos.pkl')


def test_missing_requires_explicit_fallback_without_pixel_corruption():
    fusion = dict(camera_mode='present', bypass_missing=True)
    module.configure_availability(fusion, 'missing')
    assert fusion['camera_mode'] == 'missing'
    assert module.camera_pipeline(pipeline(), 'missing', 'infos.pkl') == pipeline()
    for invalid in ({}, dict(camera_mode='present', bypass_missing=False),
                    dict(camera_mode='missing', bypass_missing=True)):
        with pytest.raises(ValueError):
            module.configure_availability(invalid, 'missing')


@pytest.mark.parametrize('mode', ['correct', 'black', 'shuffled'])
def test_corruption_does_not_enable_fallback(mode):
    fusion = dict(camera_mode='present', bypass_missing=True)
    module.configure_availability(fusion, mode)
    assert fusion['camera_mode'] == 'present'
    with pytest.raises(RuntimeError):
        module.check_availability(mode, (), dict(camera_available=torch.tensor([False])))


def test_missing_checks_every_batch_for_exact_feature_identity():
    lidar = torch.randn(1, 8, 3, 4)
    result = dict(camera_available=torch.tensor([False]), bev_features=lidar.clone())
    module.check_availability('missing', (None, lidar), result)
    result['bev_features'] = lidar + .01
    with pytest.raises(RuntimeError):
        module.check_availability('missing', (None, lidar), result)
