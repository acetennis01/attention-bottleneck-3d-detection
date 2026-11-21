"""Regression checks for label direction, gate guidance, and old checkpoints."""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from mmdet3d.structures.points import LiDARPoints


@pytest.fixture(scope='module')
def reliability_modules():
    package = ModuleType('_reliability_tests')
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'fusion')]
    transforms = ModuleType('_reliability_transforms')
    transforms.__path__ = [str(Path(__file__).resolve().parents[1] / 'transforms')]
    registry = ModuleType('mmdet3d.registry')
    registry.MODELS = registry.TRANSFORMS = SimpleNamespace(
        register_module=lambda **kwargs: lambda cls: cls)
    with patch.dict(sys.modules, {'_reliability_tests': package,
                                 '_reliability_transforms': transforms,
                                 'mmdet3d.registry': registry}):
        fusion = importlib.import_module('_reliability_tests.mbt_bev_reliability')
        transform = importlib.import_module('_reliability_transforms.sensor_reliability')
        fusion.SimpleBEVFusion = importlib.import_module(
            '_reliability_tests.simple_bev_fusion').SimpleBEVFusion
    return fusion, transform


def build(cls, **kwargs):
    return cls(in_img_channels=8, in_lidar_channels=12, embed_dim=16,
               num_heads=4, num_layers=2, num_bottleneck_tokens=4,
               token_grid_size=(2, 2), proj_dropout=0,
               point_cloud_range=(0, 0, -2, 4, 4, 1),
               height_anchors=(-1., 0.), lidar_spatial_drop_prob=0, **kwargs)


def inputs():
    meta = dict(lidar2img=torch.tensor([
        [[1., 0., 0., 0.], [0., 1., 0., 0.],
         [0., 0., 0., 1.], [0., 0., 0., 1.]]]),
        image_shapes=torch.tensor([[8, 8]]), scale_factors=torch.ones(1, 2),
        padded_image_shape=(8, 8))
    return torch.randn(1, 8, 8, 8), torch.randn(1, 12, 7, 9), meta


def test_equal_initial_scores_preserve_original_checkpoint(reliability_modules):
    mod, _ = reliability_modules
    original = build(mod.GatedCameraMBTBEVFusion, camera_drop_prob=0).eval()
    nn.init.normal_(original.bev_residual.weight, std=.02)
    guided = build(mod.ReliabilityGatedMBTBEVFusion).eval()
    result = guided.load_state_dict(original.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys and all(
        key.startswith(('camera_reliability.', 'lidar_reliability.'))
        for key in result.missing_keys)
    image, lidar, meta = inputs()
    expected = original(image, lidar, **meta)
    actual = guided(image, lidar, **meta)
    assert torch.equal(expected['bev_features'], actual['bev_features'])
    assert actual['modality_quality'].shape == (1, 2)
    guided.camera_mode = 'missing'
    assert torch.equal(guided(image, lidar, **meta)['bev_features'], lidar)


def test_lower_sensor_quality_reduces_its_gate_weight(reliability_modules):
    mod, _ = reliability_modules
    fusion = build(mod.ReliabilityGatedMBTBEVFusion)
    camera, lidar = torch.ones(2, 4, 16), torch.zeros(2, 4, 16)
    qualities = torch.tensor([[.1, .9], [.9, .1]], requires_grad=True)
    mixed = fusion._combine_bottlenecks(camera, lidar, 0, qualities)
    torch.testing.assert_close(mixed[:, 0, 0], torch.tensor([.1, .9]))
    mixed.sum().backward()
    assert torch.isfinite(qualities.grad).all() and qualities.grad.abs().sum() > 0
    fusion.reliability_strength = 0
    torch.testing.assert_close(fusion._combine_bottlenecks(
        camera, lidar, 0, qualities), torch.full_like(camera, .5))
    assert torch.isfinite(fusion._combine_bottlenecks(
        camera, lidar, 0, torch.zeros(2, 2))).all()


def test_corruption_targets_follow_the_degraded_sensor(reliability_modules):
    _, mod = reliability_modules
    image = np.full((24, 48, 3), 200, np.uint8)
    points = LiDARPoints(torch.arange(400).reshape(100, 4).float(), points_dim=4)
    original = points.tensor.clone()
    for condition, target in (('camera_dim', (.25, 1)),
                              ('camera_blur', (.25, 1)),
                              ('lidar_dropout', (1, .25)),
                              ('lidar_noise', (1, .25))):
        result, corrupted, actual = mod.corrupt_sensor_inputs(
            image, points, condition, .75, np.random.default_rng(8))
        assert actual == target
        if condition.startswith('camera_'):
            assert torch.equal(corrupted.tensor, original)
        else:
            assert np.array_equal(result, image)
        assert torch.equal(points.tensor, original)
    _, dropped, _ = mod.corrupt_sensor_inputs(
        image, points, 'lidar_dropout', 1, np.random.default_rng(8))
    assert len(dropped) == 1


def test_eval_corruption_is_deterministic_and_zero_severity_is_identity(reliability_modules):
    _, mod = reliability_modules
    image = np.full((24, 48, 3), 200, np.uint8)
    points = LiDARPoints(torch.randn(100, 4), points_dim=4)
    for condition in mod.CONDITIONS:
        transform = mod.SensorReliabilityCorruption(condition, severity=0)
        result = transform.transform(dict(img=image, points=points, sample_idx=7))
        assert np.array_equal(result['img'], image)
        assert torch.equal(result['points'].tensor, points.tensor)
    transform = mod.SensorReliabilityCorruption('lidar_noise', severity=.75)
    first = transform.transform(dict(img=image, points=points, sample_idx=7))
    second = transform.transform(dict(img=image, points=points, sample_idx=7))
    assert torch.equal(first['points'].tensor, second['points'].tensor)


@pytest.mark.parametrize('combination', ['concat', 'add'])
def test_simple_fusion_preserves_bypass_and_uses_camera(reliability_modules, combination):
    mod, _ = reliability_modules
    model = build(mod.SimpleBEVFusion, combination=combination, camera_drop_prob=0).eval()
    image, lidar, meta = inputs()
    assert torch.equal(model(image, lidar, **meta)['bev_features'], lidar)
    nn.init.normal_(model.bev_residual.weight, std=.02)
    image.requires_grad_()
    output = model(image, lidar, **meta)
    output['bev_features'].square().mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    model.camera_mode = 'missing'
    assert torch.equal(model(image, lidar, **meta)['bev_features'], lidar)
