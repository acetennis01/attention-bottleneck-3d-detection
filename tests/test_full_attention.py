"""Real-tensor checks of the unrestricted attention control (registry stub)."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from test_gated_bottlenecks import inputs


@pytest.fixture(scope='module')
def fusion_class():
    package = ModuleType('_full_attention_tests')
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'fusion')]
    registry = ModuleType('mmdet3d.registry')
    registry.MODELS = SimpleNamespace(
        register_module=lambda **kwargs: lambda cls: cls)
    with patch.dict(sys.modules, {'_full_attention_tests': package,
                                 'mmdet3d.registry': registry}):
        return importlib.import_module(
            '_full_attention_tests.full_attention_bev').FullAttentionBEVFusion


def build(cls, **kwargs):
    args = dict(in_img_channels=8, in_lidar_channels=12, embed_dim=16,
                num_heads=4, num_layers=2, token_grid_size=(2, 2),
                proj_dropout=0, point_cloud_range=(0, 0, -2, 4, 4, 1),
                height_anchors=(-1., 0.), lidar_spatial_drop_prob=0)
    args.update(kwargs)
    return cls(**args)


def test_no_bottleneck_parameters_and_zero_adapter_identity(fusion_class):
    model = build(fusion_class).eval()
    assert not any('bottleneck' in name for name in model.state_dict())
    image, lidar, meta = inputs()
    out = model(image, lidar, **meta)
    assert torch.equal(out['bev_features'], lidar)
    assert out['fusion_context_tokens'].shape == (2, 8, 16)
    assert 'bottleneck_tokens' not in out


def test_common_parameters_start_identically_to_b8(fusion_class):
    gated_class = fusion_class.__mro__[1]
    torch.manual_seed(77)
    reference = build(gated_class, num_bottleneck_tokens=8).state_dict()
    torch.manual_seed(77)
    actual = build(fusion_class).state_dict()
    assert all(torch.equal(value, reference[name])
               for name, value in actual.items())
    removed = reference.keys() - actual.keys()
    assert removed and all(name.startswith('bottleneck') for name in removed)


def test_reset_after_construction_does_not_reference_deleted_bottlenecks(fusion_class):
    model = build(fusion_class).eval()
    nn.init.normal_(model.bev_residual.weight, std=.02)
    model.reset_parameters()
    assert not any('bottleneck' in name for name in model.state_dict())
    image, lidar, meta = inputs()
    assert torch.equal(model(image, lidar, **meta)['bev_features'], lidar)


def test_full_spatial_attention_shapes_and_capture_equivalence(fusion_class):
    model = build(fusion_class).eval()
    nn.init.normal_(model.bev_residual.weight, std=.02)
    image, lidar, meta = inputs()
    plain = model(image, lidar, **meta)
    out = model(image, lidar, return_attention=True, **meta)
    torch.testing.assert_close(plain['bev_features'], out['bev_features'])
    for key in ('image_joint', 'lidar_joint'):
        assert out['attention'][key].shape == (2, 2, 4, 4, 8)
        torch.testing.assert_close(out['attention'][key].sum(-1),
                                   torch.ones(2, 2, 4, 4))
    assert out['attention']['lidar_readout'].shape == (2, 4, 4, 8)


def test_camera_reaches_lidar_in_first_layer_without_bottlenecks(fusion_class):
    torch.manual_seed(32)
    model = build(fusion_class, num_layers=1).eval()
    nn.init.normal_(model.bev_residual.weight, std=.02)
    image, lidar, meta = inputs()
    image.requires_grad_()
    first = model(image, lidar, **meta)
    second = model(image * -3, lidar, **meta)
    assert not torch.allclose(first['lidar_tokens_pre_readout'],
                              second['lidar_tokens_pre_readout'])
    first['bev_features'].square().mean().backward()
    assert image.grad.abs().sum() > 0
    for branch in (model.image_blocks, model.lidar_blocks):
        assert branch[0].attn.in_proj_weight.grad.abs().sum() > 0


def test_both_branches_read_identical_pre_layer_context(fusion_class):
    model = build(fusion_class, num_layers=1).eval()
    contexts = []
    handles = [blocks[0].register_forward_pre_hook(
        lambda module, args: contexts.append(args[1].detach().clone()))
        for blocks in (model.image_blocks, model.lidar_blocks)]
    image, lidar, meta = inputs()
    model(image, lidar, **meta)
    for handle in handles:
        handle.remove()
    assert len(contexts) == 2 and torch.equal(*contexts)


def test_missing_fallback_mixed_batch_and_dropout(fusion_class):
    model = build(fusion_class).eval()
    nn.init.normal_(model.bev_residual.weight, std=.02)
    image, lidar, meta = inputs()
    image.requires_grad_()
    out = model(image, lidar, camera_available=torch.tensor([False, True]),
                **meta)
    assert torch.equal(out['bev_features'][0], lidar[0])
    out['bev_features'].square().mean().backward()
    assert image.grad[0].count_nonzero() == 0
    assert image.grad[1].abs().sum() > 0
    model.train()
    model.camera_drop_prob = 1
    out = model(image, lidar, **meta)
    assert torch.equal(out['bev_features'], lidar)
    assert not out['image_valid_mask'].any()


def test_auxiliary_stays_camera_only_and_checkpoint_is_strict(fusion_class):
    model = build(fusion_class).eval()
    image, lidar, meta = inputs()
    image.requires_grad_()
    lidar.requires_grad_()
    out = model(image, lidar, **meta)
    out['camera_aux_logits'].square().mean().backward()
    assert image.grad.abs().sum() > 0
    assert lidar.grad is None
    assert model.image_blocks[0].attn.in_proj_weight.grad is None
    clone = build(fusion_class).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    expected = model(image, lidar, **meta)
    actual = clone(image, lidar, **meta)
    assert torch.equal(expected['bev_features'], actual['bev_features'])


@pytest.mark.parametrize('kwargs', [dict(num_bottleneck_tokens=8),
                                  dict(use_bottleneck_gate=True)])
def test_rejects_misleading_bottleneck_options(fusion_class, kwargs):
    with pytest.raises(ValueError):
        build(fusion_class, **kwargs)
