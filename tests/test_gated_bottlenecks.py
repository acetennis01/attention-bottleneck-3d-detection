"""Numerical prototype checks; registry stub only, real PyTorch operations."""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn


@pytest.fixture(scope='module')
def fusion_class():
    package = ModuleType('_gated_tests')
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'fusion')]
    registry = ModuleType('mmdet3d.registry')
    registry.MODELS = SimpleNamespace(
        register_module=lambda **kwargs: lambda cls: cls)
    with patch.dict(sys.modules, {'_gated_tests': package,
                                 'mmdet3d.registry': registry}):
        return importlib.import_module(
            '_gated_tests.mbt_bev_gated').GatedCameraMBTBEVFusion


def build(cls, **kwargs):
    args = dict(in_img_channels=8, in_lidar_channels=12, embed_dim=16,
                num_heads=4, num_layers=2, num_bottleneck_tokens=4,
                token_grid_size=(2, 2), proj_dropout=0,
                point_cloud_range=(0, 0, -2, 4, 4, 1),
                height_anchors=(-1., 0.), lidar_spatial_drop_prob=0)
    args.update(kwargs)
    return cls(**args)


def inputs(batch=2):
    meta = dict(
        lidar2img=torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.],
                                [0., 0., 0., 1.], [0., 0., 0., 1.]]]
                              ).expand(batch, -1, -1),
        image_shapes=torch.tensor([[8, 8]]).expand(batch, -1),
        scale_factors=torch.ones(batch, 2), padded_image_shape=(8, 8))
    return torch.randn(batch, 8, 8, 8), torch.randn(batch, 12, 7, 9), meta


def test_gate_initialization_matches_average_and_receives_gradients(fusion_class):
    torch.manual_seed(24)
    model = build(fusion_class).eval()
    nn.init.normal_(model.bev_residual.weight, std=.02)
    image, lidar, meta = inputs()
    image.requires_grad_()
    first = model(image, lidar, **meta)
    model.use_bottleneck_gate = False
    second = model(image, lidar, **meta)
    torch.testing.assert_close(first['bev_features'], second['bev_features'],
                               rtol=0, atol=0)
    first['bev_features'].square().mean().backward()
    assert image.grad.abs().sum() > 0
    assert all(g.weight.grad.abs().sum() > 0 for g in model.bottleneck_gates)
    assert model.bottleneck_tokens.grad.abs().sum() > 0


def test_missing_fallback_is_exact_even_with_learned_residual(fusion_class):
    model = build(fusion_class, camera_mode='missing').eval()
    nn.init.normal_(model.bev_residual.weight, std=.2)
    image, lidar, meta = inputs()
    first = model(image, lidar, **meta)
    second = model(image * 17 + 4, lidar, **meta)
    assert torch.equal(first['bev_features'], lidar)
    assert torch.equal(first['bev_features'], second['bev_features'])
    assert not first['image_valid_mask'].any()  # No auxiliary targets/loss.


def test_mixed_batch_fallback_and_camera_gradient(fusion_class):
    model = build(fusion_class).eval()
    nn.init.normal_(model.bev_residual.weight, std=.02)
    image, lidar, meta = inputs()
    image.requires_grad_()
    out = model(image, lidar, camera_available=torch.tensor([False, True]),
                **meta)
    assert torch.equal(out['bev_features'][0], lidar[0])
    assert not torch.equal(out['bev_features'][1], lidar[1])
    out['bev_features'].square().mean().backward()
    assert image.grad[0].count_nonzero() == 0
    assert image.grad[1].abs().sum() > 0


def test_dropout_applies_in_train_only(fusion_class):
    model = build(fusion_class, camera_drop_prob=1).train()
    nn.init.normal_(model.bev_residual.weight, std=.02)
    image, lidar, meta = inputs()
    out = model(image, lidar, **meta)
    assert torch.equal(out['bev_features'], lidar)
    assert not out['camera_available'].any()
    model.eval()
    out = model(image, lidar, **meta)
    assert out['camera_available'].all()
    assert not torch.equal(out['bev_features'], lidar)


def test_absent_control_is_trainable_and_image_independent(fusion_class):
    model = build(fusion_class, camera_mode='missing',
                  bypass_missing=False).eval()
    image, lidar, meta = inputs()
    image.requires_grad_()
    first = model(image, lidar, **meta)
    second = model(image * -8, lidar, **meta)
    assert torch.equal(first['bev_features'], second['bev_features'])
    first['bev_features'].square().mean().backward()
    assert model.bev_residual.weight.grad.abs().sum() > 0
    assert image.grad.count_nonzero() == 0


def test_black_image_does_not_trigger_oracle_fallback(fusion_class):
    model = build(fusion_class).eval()
    image, lidar, meta = inputs()
    assert model(torch.zeros_like(image), lidar, **meta)[
        'camera_available'].all()


def test_attention_capture_preserves_predictions(fusion_class):
    model = build(fusion_class).eval()
    nn.init.normal_(model.bev_residual.weight, std=.02)
    image, lidar, meta = inputs()
    first = model(image, lidar, **meta)
    second = model(image, lidar, return_attention=True, **meta)
    torch.testing.assert_close(first['bev_features'], second['bev_features'])


@pytest.mark.parametrize('kwargs', [dict(camera_drop_prob=-.1),
    dict(camera_drop_prob=1.1), dict(camera_mode='black'),
    dict(lidar_spatial_drop_prob=.1)])
def test_rejects_invalid_options(fusion_class, kwargs):
    with pytest.raises(ValueError):
        build(fusion_class, **kwargs)


@pytest.fixture(scope='module')
def multiview_class(fusion_class):
    package = ModuleType('_gated_tests')
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'fusion')]
    registry = ModuleType('mmdet3d.registry')
    registry.MODELS = SimpleNamespace(
        register_module=lambda **kwargs: lambda cls: cls)
    with patch.dict(sys.modules, {'_gated_tests': package,
                                 'mmdet3d.registry': registry}):
        return importlib.import_module(
            '_gated_tests.mbt_bev_gated_nuscenes'
        ).NuScenesGatedCameraMBTBEVFusion


def multi_inputs():
    image, lidar, meta = inputs()
    image = image[:, None].expand(-1, 6, -1, -1, -1).clone()
    meta['lidar2img'] = meta['lidar2img'][:, None].expand(-1, 6, -1, -1)
    meta['image_shapes'] = meta['image_shapes'][:, None].expand(-1, 6, -1)
    del meta['scale_factors']
    return image, lidar, meta


def test_six_view_dropout_excludes_weights_and_preserves_fallback(multiview_class):
    model = build(multiview_class).eval()
    nn.init.normal_(model.bev_residual.weight, std=.02)
    image, lidar, meta = multi_inputs()
    image.requires_grad_()
    views = torch.tensor([[False] * 6, [False, True, True, True, True, True]])
    out = model(image, lidar, camera_available=views, return_attention=True,
                **meta)
    weights = out['attention']['image_view_height_weights']
    assert not weights[0].any()
    assert not weights[1, 0].any()
    torch.testing.assert_close(weights[1].sum((0, 1)), torch.ones(2, 2))
    assert torch.equal(out['bev_features'][0], lidar[0])
    out['bev_features'].square().mean().backward()
    assert not image.grad[0].any()
    assert not image.grad[1, 0].any()
    assert image.grad[1, 1:].abs().sum() > 0
    assert all(g.weight.grad.abs().sum() > 0 for g in model.bottleneck_gates)


@pytest.mark.parametrize('drop', ['camera_drop_prob', 'camera_view_drop_prob'])
def test_six_view_all_drop_training_and_clean_eval(multiview_class, drop):
    model = build(multiview_class, **{drop: 1}).train()
    image, lidar, meta = multi_inputs()
    out = model(image, lidar, **meta)
    assert torch.equal(out['bev_features'], lidar)
    assert not out['image_valid_mask'].any()
    model.eval()
    assert model(image, lidar, **meta)['camera_view_available'].all()


def test_gate_weights_camera_for_positive_logit(fusion_class):
    model = build(fusion_class).eval()
    camera, lidar = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
    with torch.no_grad():
        model.bottleneck_gates[0].bias.fill_(50)
    torch.testing.assert_close(model._combine_bottlenecks(camera, lidar, 0),
                               camera)
    with torch.no_grad():
        model.bottleneck_gates[0].bias.fill_(-50)
    torch.testing.assert_close(model._combine_bottlenecks(camera, lidar, 0),
                               lidar)


def test_dropped_view_values_cannot_affect_kept_output(multiview_class):
    torch.manual_seed(41)
    model = build(multiview_class).eval()
    nn.init.normal_(model.bev_residual.weight, std=.02)
    image, lidar, meta = multi_inputs()
    views = torch.ones(2, 6, dtype=torch.bool)
    views[:, 0] = False
    first = model(image, lidar, camera_available=views, **meta)
    changed = image.clone()
    changed[:, 0] = float('nan')
    second = model(changed, lidar, camera_available=views, **meta)
    torch.testing.assert_close(first['bev_features'], second['bev_features'],
                               rtol=0, atol=0)
    assert torch.isfinite(second['bev_features']).all()


def test_whole_camera_drop_provides_no_detection_gradient(multiview_class):
    model = build(multiview_class, camera_drop_prob=1).train()
    nn.init.normal_(model.bev_residual.weight, std=.02)
    image, lidar, meta = multi_inputs()
    out = model(image, lidar, **meta)
    out['bev_features'].square().mean().backward()
    # Explicit fallback skips the learned correction; this is intended, not
    # evidence that missing examples teach the gate to recognize corruption.
    for parameter in model.parameters():
        assert parameter.grad is None or not parameter.grad.any()
