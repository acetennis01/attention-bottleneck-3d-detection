"""Real tensor tests for six-view V2 geometry, independence and masking."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from projects.myfusion.fusion.mbt_bev_nuscenes_camera_v2 import (
    NuScenesCameraFocusedMBTBEVFusionV2 as Fusion)
from projects.myfusion.fusion.mbt_detector_nuscenes import (
    NuScenesMBTBEVFusionDetector)
from projects.myfusion.fusion.mbt_detector_nuscenes_camera_v2 import (
    NuScenesCameraFocusedMBTBEVFusionDetectorV2 as Detector)


def build(**kwargs):
    args = dict(
        in_img_channels=8, in_lidar_channels=12, embed_dim=16, num_heads=4,
        num_layers=2, token_grid_size=(2, 2), num_bottleneck_tokens=8,
        point_cloud_range=(-1, -1, 0, 1, 1, 2), height_anchors=(0.5, 1.),
        camera_aux_num_classes=10, proj_dropout=0.0)
    args.update(kwargs)
    return Fusion(**args)


def inputs(invisible=False):
    projection = torch.tensor([
        [1., 0., 2., 100. if invisible else 0.],
        [0., 1., 2., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    meta = dict(
        lidar2img=projection.reshape(1, 1, 4, 4).repeat(1, 6, 1, 1),
        image_shapes=torch.full((1, 6, 2), 4.), padded_image_shape=(4, 4))
    return torch.randn(1, 6, 8, 4, 4), torch.randn(1, 12, 7, 9), meta


@pytest.mark.parametrize('tokens', [8, 16])
def test_multiview_shapes_normalization_and_baseline_identity(tokens):
    module = build(num_bottleneck_tokens=tokens).eval()
    image, lidar, meta = inputs()
    with patch.object(module.aligned_camera_norm, 'forward',
                      wraps=module.aligned_camera_norm.forward) as norm:
        output = module(image, lidar, **meta, return_attention=True)
    assert norm.call_count == 1
    torch.testing.assert_close(output['bev_features'], lidar, rtol=0, atol=0)
    assert output['bottleneck_tokens'].shape == (1, tokens, 16)
    assert output['camera_aux_logits'].shape == (1, 4, 10)
    weights = output['attention']['image_view_height_weights']
    assert weights.shape == (1, 6, 2, 2, 2)
    torch.testing.assert_close(weights.sum((1, 2)), torch.ones(1, 2, 2))
    assert float(output['camera_weight_sum_min']) == pytest.approx(1)
    assert float(output['camera_weight_sum_max']) == pytest.approx(1)


def test_auxiliary_has_no_lidar_or_bottleneck_dependency():
    module = build().eval()
    image, lidar, meta = inputs()
    image.requires_grad_()
    lidar.requires_grad_()
    output = module(image, lidar, **meta)
    output['camera_aux_logits'].square().mean().backward()
    assert image.grad.abs().sum() > 0
    assert lidar.grad is None
    assert module.bottleneck_tokens.grad is None
    assert module.image_blocks[0].attn.in_proj_weight.grad is None


def test_detection_objective_reaches_all_valid_camera_views_through_mbt():
    module = build().eval()
    nn.init.normal_(module.bev_residual.weight, std=0.02)
    image, lidar, meta = inputs()
    image.requires_grad_()
    output = module(image, lidar, **meta)
    output['bev_features'].square().mean().backward()
    assert (image.grad.abs().flatten(2).sum(2) > 0).all()
    assert module.bottleneck_tokens.grad.abs().sum() > 0
    assert module.camera_aux_head.weight.grad is None


def test_invalid_views_have_zero_weight_and_no_invalid_camera_gradient():
    module = build().eval()
    image, lidar, meta = inputs()
    meta['lidar2img'][:, 1:, 0, 3] = 100
    image.requires_grad_()
    output = module(image, lidar, **meta, return_attention=True)
    weights = output['attention']['image_view_height_weights']
    assert not weights[:, 1:].any()
    torch.testing.assert_close(weights[:, 0].sum(1), torch.ones(1, 2, 2))
    output['camera_aux_logits'].sum().backward()
    assert image.grad[:, 0].abs().sum() > 0
    assert not image.grad[:, 1:].any()


def test_all_invalid_is_finite_and_does_not_mask_lidar():
    module = build(lidar_spatial_drop_prob=0.99).train()
    image, lidar, meta = inputs(invisible=True)
    output = module(image, lidar, **meta, return_attention=True)
    assert not output['image_valid_mask'].any()
    assert output['lidar_token_keep_mask'].all()
    assert not output['attention']['image_view_height_weights'].any()
    assert torch.isfinite(output['camera_aux_logits']).all()
    torch.testing.assert_close(output['bev_features'], lidar, rtol=0, atol=0)


def test_multiview_feature_extraction_keeps_batch_and_view_order():
    model = Detector.__new__(Detector)
    nn.Module.__init__(model)

    class Backbone(nn.Module):
        def forward(self, image):
            return (image, image * 2, image * 3)

    class Neck(nn.Module):
        def forward(self, features):
            return (sum(features),)

    model.img_backbone, model.img_neck = Backbone(), Neck()
    model.img_feature_level = 0
    image = torch.randn(2, 6, 3, 4, 4)
    torch.testing.assert_close(model.extract_img_feat(image), image * 6)


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('drop_probability', [0.0, 0.5])
def test_same_keep_mask_covers_every_lidar_pyramid_bypass(
        training, drop_probability):
    model = Detector.__new__(Detector)
    nn.Module.__init__(model)
    model.fusion_module = build(lidar_spatial_drop_prob=drop_probability)
    model.fusion_level = 0
    model.train(training)
    features = (torch.ones(1, 12, 7, 9), torch.ones(1, 12, 3, 5),
                torch.ones(1, 12, 2, 2))
    keep = torch.tensor([[[False], [True], [True], [False]]])
    output = dict(lidar_token_keep_mask=keep)
    with patch.object(NuScenesMBTBEVFusionDetector, 'extract_feat',
                      return_value=(features, output)):
        result = model.extract_feat({})
    # The selected fusion level is already processed and must not be re-masked.
    torch.testing.assert_close(result[0], features[0])
    for feature in result[1:]:
        if training and drop_probability > 0:
            assert feature[0, 0, 0, 0] == 0
            assert feature[0, 0, -1, -1] == 0
            assert feature[0, 0, 0, -1] == 1
            assert feature[0, 0, -1, 0] == 1
        else:
            assert (feature == 1).all()


def test_training_zero_mask_policy_preserves_lidar_and_draws_no_mask():
    module = build(lidar_spatial_drop_prob=0.0).train()
    image, lidar, meta = inputs()
    with patch('torch.rand_like', side_effect=AssertionError('Mask was sampled')):
        output = module(image, lidar, **meta)
    assert output['lidar_token_keep_mask'].all()
    torch.testing.assert_close(output['bev_features'], lidar, rtol=0, atol=0)


def test_six_views_and_matching_metadata_are_required():
    module = build().eval()
    image, lidar, meta = inputs()
    with pytest.raises(ValueError, match='six-view'):
        module(image[:, :1], lidar, **meta)
    with pytest.raises(ValueError, match='calibration'):
        module(image, lidar)
    meta['image_shapes'] = torch.ones(1, 2)
    with pytest.raises(ValueError, match='image_shapes'):
        module(image, lidar, **meta)
