"""Focused tests for the symmetric camera-LiDAR MBT fusion module."""

import torch

from projects.myfusion.fusion.mbt_bev import SymmetricMBTBEVFusion


def _build(num_layers=2):
    return SymmetricMBTBEVFusion(
        in_img_channels=8,
        in_lidar_channels=12,
        embed_dim=16,
        num_heads=4,
        mlp_ratio=2,
        token_grid_size=(4, 5),
        num_bottleneck_tokens=4,
        num_layers=num_layers,
        proj_dropout=0.0,
    ).eval()


def test_shapes_initialization_and_baseline_identity():
    torch.manual_seed(0)
    module = _build(num_layers=2)
    image = torch.randn(2, 8, 10, 14)
    lidar = torch.randn(2, 12, 16, 18)
    output = module(image, lidar)

    assert output['bev_features'].shape == lidar.shape
    assert output['image_tokens'].shape == (2, 20, 16)
    assert output['lidar_tokens'].shape == (2, 20, 16)
    assert output['lidar_tokens_pre_readout'].shape == (2, 20, 16)
    assert output['bottleneck_tokens'].shape == (2, 4, 16)
    torch.testing.assert_close(output['bev_features'], lidar)
    assert abs(float(module.bottleneck_tokens.mean())) < 0.02
    assert 0.01 < float(module.bottleneck_tokens.std()) < 0.03


def test_cross_modal_information_only_passes_through_bottlenecks():
    torch.manual_seed(1)
    lidar = torch.randn(1, 12, 8, 9)
    image_a = torch.randn(1, 8, 7, 11)
    image_b = image_a + 2.0

    module = _build(num_layers=1)
    output_a = module(image_a, lidar)
    output_b = module(image_b, lidar)

    # The simultaneous symmetric update has no direct image-to-LiDAR edge.
    torch.testing.assert_close(
        output_a['lidar_tokens_pre_readout'],
        output_b['lidar_tokens_pre_readout'],
        atol=1e-6,
        rtol=1e-6,
    )
    # The detection-specific readout then transfers the image-conditioned
    # bottlenecks into the LiDAR representation.
    assert not torch.allclose(
        output_a['lidar_tokens'], output_b['lidar_tokens'],
        atol=1e-6, rtol=1e-6)


def test_modalities_have_separate_parameters_and_receive_gradients():
    torch.manual_seed(2)
    module = _build(num_layers=2)
    assert (module.image_blocks[0].attn.in_proj_weight is not
            module.lidar_blocks[0].attn.in_proj_weight)

    # Enable the residual adapter so the detection representation exercises
    # both modality branches in this isolated gradient test.
    torch.nn.init.normal_(module.bev_residual.weight, std=0.01)
    image = torch.randn(2, 8, 9, 10, requires_grad=True)
    lidar = torch.randn(2, 12, 11, 12, requires_grad=True)
    loss = module(image, lidar)['bev_features'].square().mean()
    loss.backward()

    assert image.grad is not None and image.grad.abs().sum() > 0
    assert lidar.grad is not None and lidar.grad.abs().sum() > 0
    assert module.bottleneck_tokens.grad is not None
    assert module.bottleneck_tokens.grad.abs().sum() > 0
