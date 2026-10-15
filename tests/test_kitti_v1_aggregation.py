"""PyTorch-only regressions for the historical KITTI aggregation correction.

Framework registration and the diagnostic hook's base class are stubbed; all
model, attention, sampling and autograd operations use real PyTorch.
MMDetection3D/CUDA are not required for these numerical unit tests.
"""

import importlib.util
from pathlib import Path
import runpy
import sys
import types
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def fusion_class():
    registry = types.ModuleType('mmdet3d.registry')

    class Registry:
        def register_module(self, **kwargs):
            return lambda cls: cls

    registry.MODELS = Registry()
    spec = importlib.util.spec_from_file_location(
        '_kitti_v1_under_test', ROOT / 'fusion/mbt_bev_kitti_v1.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
            'mmdet3d': types.ModuleType('mmdet3d'),
            'mmdet3d.registry': registry}):
        spec.loader.exec_module(module)
    return module.SymmetricMBTBEVFusion


def build(cls, mode='valid_only', tokens=4):
    return cls(
        in_img_channels=8, in_lidar_channels=12,
        embed_dim=16, num_heads=4, num_layers=2,
        token_grid_size=(2, 2), num_bottleneck_tokens=tokens,
        point_cloud_range=(0, 0, -2, 4, 4, 1),
        height_anchors=(-1., 0.), use_local_camera_residual=False,
        camera_aux_num_classes=3, camera_aggregation_mode=mode,
    ).eval()


def metadata(invisible=False):
    return dict(
        lidar2img=torch.tensor([[
            [1., 0., 0., 100. if invisible else 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
        ]]),
        image_shapes=torch.tensor([[8, 8]]),
        scale_factors=torch.ones(1, 2), padded_image_shape=(8, 8))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_extreme_scores_keep_valid_weights_and_gradients(fusion_class, dtype):
    module = build(fusion_class)
    # The last height candidate is invalid but would win the old -10000 mask.
    scores = torch.tensor([-20000., -30000., 0.], dtype=dtype).reshape(1, 3, 1, 1)
    scores.requires_grad_()
    valid = torch.tensor([True, True, False]).reshape_as(scores)
    weights = module._camera_candidate_weights(scores, valid)
    torch.testing.assert_close(weights.sum(1), torch.ones(1, 1, 1, dtype=dtype))
    assert weights[0, 0, 0, 0] == 1
    assert weights[0, 2, 0, 0] == 0
    (weights * torch.tensor([1., 2., 3.]).reshape_as(weights)).sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[0, 2, 0, 0] == 0


def test_all_invisible_candidates_have_zero_weights_and_gradients(fusion_class):
    module = build(fusion_class)
    scores = torch.full((2, 4, 2, 2), -50000., requires_grad=True)
    weights = module._camera_candidate_weights(scores, torch.zeros_like(scores).bool())
    assert torch.isfinite(weights).all() and not weights.any()
    weights.sum().backward()
    assert torch.isfinite(scores.grad).all() and not scores.grad.any()


def test_normal_scores_and_gradients_match_historical_arithmetic(fusion_class):
    legacy = build(fusion_class, mode='legacy')
    stable = build(fusion_class)
    torch.manual_seed(17)
    first = torch.randn(2, 4, 3, 5, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    valid = torch.rand_like(first) > 0.3
    coefficients = torch.randn_like(first)
    old = legacy._camera_candidate_weights(first, valid)
    new = stable._camera_candidate_weights(second, valid)
    torch.testing.assert_close(old, new, atol=1e-7, rtol=1e-6)
    (old * coefficients).sum().backward()
    (new * coefficients).sum().backward()
    torch.testing.assert_close(first.grad, second.grad, atol=1e-7, rtol=1e-5)


def test_legacy_mode_reproduces_old_failure(fusion_class):
    module = build(fusion_class, mode='legacy')
    scores = torch.tensor([-20000., -30000., 0.]).reshape(1, 3, 1, 1)
    valid = torch.tensor([True, True, False]).reshape_as(scores)
    assert not module._camera_candidate_weights(scores, valid).any()


def test_legacy_arithmetic_is_bitwise_identical(fusion_class):
    module = build(fusion_class, mode='legacy')
    torch.manual_seed(19)
    scores = torch.randn(2, 4, 2, 3)
    valid = torch.rand_like(scores) > .3
    original = torch.softmax(scores.masked_fill(~valid, -1e4), dim=1).unsqueeze(1)
    original = original * valid.unsqueeze(1).to(original.dtype)
    original = original / original.sum(dim=2, keepdim=True).clamp_min(1e-6)
    actual = module._camera_candidate_weights(scores, valid).unsqueeze(1)
    assert torch.equal(original, actual)


def test_legacy_is_default_and_unknown_mode_is_rejected(fusion_class):
    module = fusion_class(in_img_channels=8, in_lidar_channels=12,
                          embed_dim=16, num_heads=4)
    assert module.camera_aggregation_mode == 'legacy'
    with pytest.raises(ValueError, match='camera_aggregation_mode'):
        build(fusion_class, mode='typo')


def test_modes_keep_checkpoint_keys_and_normal_outputs(fusion_class):
    torch.manual_seed(23)
    legacy = build(fusion_class, mode='legacy')
    stable = build(fusion_class)
    stable.load_state_dict(legacy.state_dict(), strict=True)
    torch.nn.init.normal_(legacy.bev_residual.weight, std=.01)
    stable.load_state_dict(legacy.state_dict(), strict=True)
    assert legacy.state_dict().keys() == stable.state_dict().keys()
    image, lidar = torch.randn(1, 8, 8, 8), torch.randn(1, 12, 6, 6)
    old = legacy(image, lidar, **metadata())
    new = stable(image, lidar, **metadata())
    torch.testing.assert_close(old['bev_features'], new['bev_features'])
    torch.testing.assert_close(old['camera_aux_logits'], new['camera_aux_logits'])
    assert float(new['camera_weight_sum_min']) == pytest.approx(1.)
    assert float(new['camera_weight_sum_max']) == pytest.approx(1.)


def test_health_check_exposes_collapse_and_handles_invisible_frames(fusion_class):
    module = build(fusion_class, mode='legacy')
    # A tiny image makes one height visible and the other invisible. Force
    # valid logits below the historical sentinel to exercise the real sampler.
    with torch.no_grad():
        module.height_score.weight.zero_()
        module.height_score.bias.fill_(-20000.)
    meta = metadata()
    meta['lidar2img'][0, 1] = torch.tensor([0., 0., 10., 10.])
    image, lidar = torch.randn(1, 8, 8, 8), torch.randn(1, 12, 6, 6)
    old = module(image, lidar, **meta)
    assert old['image_valid_mask'].any()
    assert float(old['camera_weight_sum_min']) == 0.
    module.camera_aggregation_mode = 'valid_only'
    new = module(image, lidar, **meta)
    assert float(new['camera_weight_sum_min']) == pytest.approx(1.)
    invisible = module(image, lidar, **metadata(invisible=True))
    assert not invisible['image_valid_mask'].any()
    assert torch.isfinite(invisible['bev_features']).all()


def test_mixed_batch_health_excludes_invisible_cells(fusion_class):
    module = build(fusion_class)
    visible = metadata()
    hidden = metadata(invisible=True)
    meta = {key: torch.cat((visible[key], hidden[key]), dim=0)
            for key in ('lidar2img', 'image_shapes', 'scale_factors')}
    meta['padded_image_shape'] = visible['padded_image_shape']
    output = module(torch.randn(2, 8, 8, 8), torch.randn(2, 12, 6, 6), **meta)
    assert output['image_valid_mask'][0].all()
    assert not output['image_valid_mask'][1].any()
    assert float(output['camera_weight_sum_min']) == pytest.approx(1.)
    assert float(output['camera_weight_sum_max']) == pytest.approx(1.)


@pytest.mark.parametrize('minimum,maximum', [(0., 1.), (1., 1.1), (1., 1.)])
def test_diagnostic_hook_checks_both_weight_bounds(minimum, maximum):
    registry = types.ModuleType('mmdet3d.registry')
    registry.HOOKS = types.SimpleNamespace(
        register_module=lambda: lambda cls: cls)
    hooks = types.ModuleType('mmengine.hooks')
    hooks.Hook = type('Hook', (), {})
    model = types.ModuleType('mmengine.model')
    model.is_model_wrapper = lambda instance: False
    spec = importlib.util.spec_from_file_location(
        '_diagnostics_under_test', ROOT / 'hooks/fusion_diagnostics.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
            'mmdet3d': types.ModuleType('mmdet3d'),
            'mmdet3d.registry': registry,
            'mmengine': types.ModuleType('mmengine'),
            'mmengine.hooks': hooks, 'mmengine.model': model}):
        spec.loader.exec_module(module)
    hook = module.FusionDiagnosticsHook(interval=1)
    hook._is_logging_step = lambda: True
    feature = torch.ones(1, 2, 2, 2)
    output = dict(bev_features=feature, image_tokens=feature,
                  lidar_tokens=feature, bottleneck_tokens=feature,
                  camera_weight_sum_min=torch.tensor(minimum),
                  camera_weight_sum_max=torch.tensor(maximum))
    if minimum < .999 or maximum > 1.001:
        with pytest.raises(RuntimeError, match='Camera aggregation'):
            hook._forward_hook(None, (feature, feature), output)
    else:
        hook._forward_hook(None, (feature, feature), output)
        assert hook._feature_stats['fusion/camera_weight_sum_min'] == 1.
        assert hook._feature_stats['fusion/camera_weight_sum_max'] == 1.


@pytest.mark.parametrize('tokens', [1, 2, 4, 8, 16])
def test_all_token_counts_retain_bottleneck_conditioned_auxiliary(fusion_class, tokens):
    torch.manual_seed(29)
    module = build(fusion_class, tokens=tokens)
    image = torch.randn(1, 8, 8, 8, requires_grad=True)
    lidar = torch.randn(1, 12, 6, 6, requires_grad=True)
    output = module(image, lidar, return_attention=True, **metadata())
    assert output['bottleneck_tokens'].shape == (1, tokens, 16)
    assert output['bev_features'].shape == lidar.shape
    assert 'camera_readout' in output['attention']
    output['camera_aux_logits'].square().mean().backward()
    for gradient in (image.grad, lidar.grad, module.bottleneck_tokens.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_variant_configs_share_recipe_and_keep_predictions_separate():
    base = runpy.run_path(str(ROOT / 'configs/kitti_bottleneck_base.py'))
    assert base['train_cfg']['max_epochs'] == 40
    assert base['custom_hooks'][0]['freeze_epochs'] == 5
    assert base['optim_wrapper']['accumulative_counts'] == 8
    assert base['model']['fusion_module']['camera_aggregation_mode'] == 'valid_only'
    paths = set()
    for tokens in (1, 2, 4, 8, 16):
        cfg = runpy.run_path(str(ROOT / f'configs/kitti_bottleneck_b{tokens}.py'))
        assert cfg['_base_'] == ['./kitti_bottleneck_base.py']
        assert cfg['model'] == dict(fusion_module=dict(num_bottleneck_tokens=tokens))
        assert cfg['work_dir'] not in paths
        paths.add(cfg['work_dir'])
        assert cfg['val_evaluator']['pklfile_prefix'].startswith(cfg['work_dir'])
        assert cfg['test_evaluator']['pklfile_prefix'].startswith(cfg['work_dir'])
