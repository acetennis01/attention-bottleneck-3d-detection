"""Real-PyTorch graph and masking tests; registration is the only stub.

These tests do not require MMDetection3D, compiled CUDA ops, or a dataset.
End-to-end MMDetection3D checks are separate from these numerical regressions.
"""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def components():
    package = ModuleType('_camera_v2_tests')
    package.__path__ = [str(ROOT / 'fusion')]
    registry = ModuleType('mmdet3d.registry')
    registry.MODELS = SimpleNamespace(
        register_module=lambda **kwargs: lambda cls: cls)
    models = ModuleType('mmdet3d.models')
    models.Base3DDetector = nn.Module
    with patch.dict(sys.modules, {
            '_camera_v2_tests': package,
            'mmdet3d': ModuleType('mmdet3d'),
            'mmdet3d.registry': registry,
            'mmdet3d.models': models}):
        fusion = importlib.import_module(
            '_camera_v2_tests.mbt_bev_camera_v2')
        detector = importlib.import_module(
            '_camera_v2_tests.mbt_detector_camera_v2')
    return (fusion.CameraFocusedMBTBEVFusionV2,
            detector.CameraFocusedMBTBEVFusionDetectorV2)


def build(components, **kwargs):
    args = dict(
        in_img_channels=8, in_lidar_channels=12, embed_dim=16, num_heads=4,
        num_layers=2, token_grid_size=(2, 2), num_bottleneck_tokens=8,
        point_cloud_range=(0, 0, -2, 4, 4, 1), height_anchors=(-1., 0.),
        camera_aux_num_classes=3, lidar_spatial_drop_prob=0.1,
        proj_dropout=0.0)
    args.update(kwargs)
    return components[0](**args)


def metadata(invisible=False):
    return dict(
        lidar2img=torch.tensor([[
            [1., 0., 0., 100. if invisible else 0.],
            [0., 1., 0., 0.], [0., 0., 0., 1.], [0., 0., 0., 1.]]]),
        image_shapes=torch.tensor([[8, 8]]),
        scale_factors=torch.ones(1, 2), padded_image_shape=(8, 8))


def inputs():
    return torch.randn(1, 8, 8, 8), torch.randn(1, 12, 7, 9)


def test_eval_starts_exactly_at_lidar_baseline(components):
    module = build(components).eval()
    image, lidar = inputs()
    output = module(image, lidar, **metadata())
    torch.testing.assert_close(output['bev_features'], lidar, rtol=0, atol=0)
    assert output['bottleneck_tokens'].shape == (1, 8, 16)
    assert output['camera_aux_logits'].shape == (1, 4, 3)
    assert output['lidar_token_keep_mask'].all()
    assert not hasattr(module, 'camera_readout')
    assert not hasattr(module, 'local_gate')


def test_auxiliary_is_independent_of_lidar_and_bottlenecks(components):
    torch.manual_seed(21)
    module = build(components).eval()
    image, lidar = inputs()
    image.requires_grad_()
    lidar.requires_grad_()
    first = module(image, lidar, **metadata())
    with torch.no_grad():
        module.bottleneck_tokens.add_(torch.randn_like(module.bottleneck_tokens))
    second = module(image, lidar * 10 + 5, **metadata())
    torch.testing.assert_close(first['camera_aux_logits'],
                               second['camera_aux_logits'], rtol=0, atol=0)
    # Use a new forward after modifying the parameter to avoid stale autograd.
    second['camera_aux_logits'].square().mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert lidar.grad is None
    assert module.bottleneck_tokens.grad is None
    assert module.lidar_projection.weight.grad is None
    assert module.image_blocks[0].attn.in_proj_weight.grad is None


def test_coupled_mask_hides_bypass_and_pooled_content(components):
    module = build(components, lidar_spatial_drop_prob=0.5).train()
    # First and last cells are dropped; one requested drop is out of camera FOV.
    valid = torch.tensor([[[True], [False], [True], [True]]])
    draws = torch.tensor([[[0.0], [0.0], [1.0], [0.0]]])
    lidar = torch.ones(1, 12, 7, 9)
    with patch('torch.rand_like', return_value=draws):
        content, bypass, keep = module._mask_lidar(lidar, valid)
    assert keep.tolist() == [[[False], [True], [True], [False]]]
    assert not content[:, [0, 3]].any()  # Re-mask after non-divisible pooling.
    assert not bypass[:, :, :4, :5].any()
    assert not bypass[:, :, 4:, 5:].any()
    assert (bypass[:, :, :4, 5:] == 1).all()  # No inverted-dropout scaling.
    assert (bypass[:, :, 4:, :5] == 1).all()


def test_forward_uses_the_same_mask_for_residual_bypass(components):
    module = build(components, lidar_spatial_drop_prob=0.5).train()
    image, lidar = inputs()
    draws = torch.tensor([[[0.0], [1.0], [1.0], [0.0]]])
    with patch('torch.rand_like', return_value=draws):
        output = module(image, lidar, **metadata())
    # At zero adapter initialization, output must be precisely the masked bypass.
    expected = lidar.clone()
    expected[:, :, :4, :5] = 0
    expected[:, :, 4:, 5:] = 0
    torch.testing.assert_close(output['bev_features'], expected, rtol=0, atol=0)


def test_eval_disables_masking_and_is_deterministic(components):
    module = build(components, lidar_spatial_drop_prob=0.99).eval()
    image, lidar = inputs()
    with patch('torch.rand_like', side_effect=AssertionError('eval drew a mask')):
        first = module(image, lidar, **metadata())
        second = module(image, lidar, **metadata())
    torch.testing.assert_close(first['bev_features'], second['bev_features'])
    assert first['lidar_token_keep_mask'].all()


def test_invisible_camera_cells_are_not_masked(components):
    module = build(components, lidar_spatial_drop_prob=0.99).train()
    image, lidar = inputs()
    output = module(image, lidar, **metadata(invisible=True))
    assert not output['image_valid_mask'].any()
    assert output['lidar_token_keep_mask'].all()
    assert torch.isfinite(output['bev_features']).all()
    torch.testing.assert_close(output['bev_features'], lidar)
    assert float(output['camera_weight_sum_min']) == 1


def test_camera_to_lidar_exchange_remains_bottleneck_mediated(components):
    torch.manual_seed(22)
    module = build(components, num_layers=1).eval()
    nn.init.normal_(module.bev_residual.weight, std=0.02)
    image, lidar = inputs()
    other = torch.randn_like(image)
    first = module(image, lidar, **metadata())
    second = module(other, lidar, **metadata())
    torch.testing.assert_close(first['lidar_tokens_pre_readout'],
                               second['lidar_tokens_pre_readout'])
    assert not torch.allclose(first['lidar_tokens'], second['lidar_tokens'])
    # Removing the shared state removes the camera->LiDAR route entirely.
    handle = module.bottleneck_norm.register_forward_hook(
        lambda _module, _args, result: torch.zeros_like(result))
    try:
        first = module(image, lidar, **metadata())
        second = module(other, lidar, **metadata())
    finally:
        handle.remove()
    torch.testing.assert_close(first['bev_features'], second['bev_features'])


def test_detection_gradients_reach_camera_and_bottlenecks(components):
    torch.manual_seed(23)
    module = build(components).eval()
    nn.init.normal_(module.bev_residual.weight, std=0.02)
    image, lidar = inputs()
    image.requires_grad_()
    module(image, lidar, **metadata())['bev_features'].square().mean().backward()
    assert image.grad.abs().sum() > 0
    assert module.bottleneck_tokens.grad.abs().sum() > 0
    assert module.image_blocks[0].attn.in_proj_weight.grad.abs().sum() > 0


def test_attention_capture_does_not_change_predictions(components):
    module = build(components).eval()
    image, lidar = inputs()
    ordinary = module(image, lidar, **metadata())
    captured = module(image, lidar, return_attention=True, **metadata())
    torch.testing.assert_close(ordinary['bev_features'], captured['bev_features'])
    assert captured['attention']['image_bottleneck'].shape == (1, 2, 4, 8, 4)
    assert captured['attention']['lidar_readout'].shape == (1, 4, 4, 8)
    assert captured['attention']['image_height_weights'].shape == (1, 2, 2, 2)
    weights = captured['attention']['image_height_weights']
    torch.testing.assert_close(weights.sum(1), torch.ones(1, 2, 2))


@pytest.mark.parametrize('kwargs', [
    dict(lidar_spatial_drop_prob=1), dict(lidar_spatial_drop_prob=-0.1),
    dict(camera_aux_num_classes=0), dict(height_score_limit=0),
    dict(use_local_camera_residual=True), dict(lidar_token_drop_prob=0.1),
    dict(lidar_bev_drop_prob=0.1), dict(camera_aggregation_mode='legacy'),
    dict(point_cloud_range=None),
])
def test_rejects_unsafe_or_inconsistent_options(components, kwargs):
    with pytest.raises(ValueError):
        build(components, **kwargs)


def test_image_neck_selects_stride8_pyramid_level(components):
    cls = components[1]
    model = cls.__new__(cls)
    nn.Module.__init__(model)

    class Backbone(nn.Module):
        def forward(self, image):
            return (image * 2, image * 3, image * 4)

    class Neck(nn.Module):
        def forward(self, features):
            assert len(features) == 3
            return (sum(features), features[1], features[2])

    model.img_backbone, model.img_neck = Backbone(), Neck()
    model.img_feature_level = 0
    image = torch.randn(1, 3, 4, 4, requires_grad=True)
    output = model.extract_img_feat(image)
    torch.testing.assert_close(output, image * 9)
    output.sum().backward()
    torch.testing.assert_close(image.grad, torch.full_like(image, 9))
    model.img_feature_level = 3
    with pytest.raises(ValueError):
        model.extract_img_feat(image)


def test_mask_schedule_is_resume_safe():
    import importlib.util

    hooks = ModuleType('mmengine.hooks')
    hooks.Hook = object
    models = ModuleType('mmengine.model')
    models.is_model_wrapper = lambda model: False
    registry = ModuleType('mmdet3d.registry')
    registry.HOOKS = SimpleNamespace(
        register_module=lambda **kwargs: lambda cls: cls)
    spec = importlib.util.spec_from_file_location(
        '_camera_v2_masking_tests', ROOT / 'hooks/camera_v2_masking.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
            'mmengine.hooks': hooks, 'mmengine.model': models,
            'mmdet3d.registry': registry}):
        spec.loader.exec_module(module)
    hook = module.CameraV2MaskingHook()
    fusion = SimpleNamespace(lidar_spatial_drop_prob=0.1)
    runner = SimpleNamespace(model=SimpleNamespace(fusion_module=fusion),
                             logger=SimpleNamespace(info=lambda text: None))
    # Non-monotonic order simulates resuming directly in a later epoch.
    for epoch, expected in [(0, 0), (1, 0), (2, 0.1 / 3),
                            (4, 0.1), (3, 0.2 / 3), (30, 0.1)]:
        runner.epoch = epoch
        hook.before_train_epoch(runner)
        assert fusion.lidar_spatial_drop_prob == pytest.approx(expected)
