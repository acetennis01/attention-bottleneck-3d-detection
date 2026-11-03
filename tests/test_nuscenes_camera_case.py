"""Camera corruption must not change the detector or LiDAR pipeline."""
import importlib.util
from pathlib import Path
import pytest

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


@pytest.mark.parametrize('mode', ['correct', 'black', 'shuffled'])
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


@pytest.mark.parametrize('mode', ['correct', 'black', 'shuffled'])
def test_requires_camera_loader(mode):
    with pytest.raises(ValueError):
        module.camera_pipeline([dict(type='LoadPointsFromFile')], mode, 'infos.pkl')


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        module.camera_pipeline(pipeline(), 'missing', 'infos.pkl')
