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


def _build_aligned():
    return SymmetricMBTBEVFusion(
        in_img_channels=8,
        in_lidar_channels=12,
        embed_dim=16,
        num_heads=4,
        mlp_ratio=2,
        token_grid_size=(2, 2),
        num_bottleneck_tokens=4,
        num_layers=2,
        proj_dropout=0.0,
        point_cloud_range=(0, 0, -2, 4, 4, 1),
        height_anchors=(-1.0, 0.0),
    ).eval()


def _build_camera_focused():
    return SymmetricMBTBEVFusion(
        in_img_channels=8,
        in_lidar_channels=12,
        embed_dim=16,
        num_heads=4,
        mlp_ratio=2,
        token_grid_size=(2, 2),
        num_bottleneck_tokens=4,
        num_layers=2,
        proj_dropout=0.0,
        point_cloud_range=(0, 0, -2, 4, 4, 1),
        height_anchors=(-1.0, 0.0),
        use_local_camera_residual=False,
        lidar_token_drop_prob=0.9,
        camera_aux_num_classes=3,
    )


def _projection_metadata(batch_size=1):
    # Homogeneous projection with depth fixed to one: u=x and v=y.
    lidar2img = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]).unsqueeze(0).repeat(batch_size, 1, 1)
    return dict(
        lidar2img=lidar2img,
        image_shapes=torch.tensor([[8, 8]]).repeat(batch_size, 1),
        scale_factors=torch.ones(batch_size, 2),
        padded_image_shape=(8, 8),
    )


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


def test_geometry_aligned_projection_and_validity_mask():
    torch.manual_seed(3)
    module = _build_aligned()
    image = torch.randn(1, 8, 8, 8)
    lidar = torch.randn(1, 12, 6, 6)
    output = module(image, lidar, **_projection_metadata())

    assert output['image_tokens'].shape == (1, 4, 16)
    assert output['image_valid_mask'].shape == (1, 4, 1)
    assert output['image_valid_mask'].all()
    assert float(output['image_valid_ratio']) == 1.0
    assert output['local_camera_gate'].shape == (1, 4, 16)
    # Zero initialization still begins exactly at the LiDAR baseline.
    torch.testing.assert_close(output['bev_features'], lidar)


def test_geometry_alignment_changes_when_calibration_changes():
    torch.manual_seed(4)
    module = _build_aligned()
    image = torch.randn(1, 8, 8, 8)
    lidar = torch.randn(1, 12, 6, 6)
    valid_meta = _projection_metadata()
    invalid_meta = _projection_metadata()
    invalid_meta['lidar2img'][:, 0, 3] = 100.0

    valid = module(image, lidar, **valid_meta)
    invalid = module(image, lidar, **invalid_meta)

    assert valid['image_valid_mask'].all()
    assert not invalid['image_valid_mask'].any()
    assert not torch.allclose(
        valid['lidar_tokens_pre_readout'],
        invalid['lidar_tokens_pre_readout'])


def test_camera_focused_path_is_bottleneck_mediated_and_supervised():
    torch.manual_seed(5)
    module = _build_camera_focused()
    image = torch.randn(1, 8, 8, 8, requires_grad=True)
    lidar = torch.randn(1, 12, 6, 6, requires_grad=True)
    output = module(image, lidar, **_projection_metadata())

    assert 'local_camera_gate' not in output
    assert output['camera_aux_logits'].shape == (1, 4, 3)
    assert output['camera_aux_tokens'].shape == (1, 4, 16)
    assert not output['lidar_token_keep_mask'].all()
    # The zero-initialized detector adapter still protects the LiDAR baseline.
    torch.testing.assert_close(output['bev_features'], lidar)

    auxiliary_loss = output['camera_aux_logits'].square().mean()
    auxiliary_loss.backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert module.bottleneck_tokens.grad is not None
    assert module.bottleneck_tokens.grad.abs().sum() > 0
    assert module.camera_readout.in_proj_weight.grad is not None


def test_camera_focused_eval_disables_lidar_dropout():
    torch.manual_seed(6)
    module = _build_camera_focused().eval()
    output = module(
        torch.randn(1, 8, 8, 8),
        torch.randn(1, 12, 6, 6),
        **_projection_metadata())
    assert output['lidar_token_keep_mask'].all()
    assert float(output['lidar_token_keep_ratio']) == 1.0


def test_attention_capture_is_opt_in_and_preserves_output():
    torch.manual_seed(7)
    module = _build_camera_focused().eval()
    image = torch.randn(1, 8, 8, 8)
    lidar = torch.randn(1, 12, 6, 6)
    metadata = _projection_metadata()

    ordinary = module(image, lidar, **metadata)
    captured = module(
        image, lidar, return_attention=True, **metadata)

    assert 'attention' not in ordinary
    torch.testing.assert_close(
        captured['bev_features'], ordinary['bev_features'],
        atol=1e-6, rtol=1e-5)
    attention = captured['attention']
    assert attention['image_bottleneck'].shape == (1, 2, 4, 4, 4)
    assert attention['lidar_bottleneck'].shape == (1, 2, 4, 4, 4)
    assert attention['lidar_readout'].shape == (1, 4, 4, 4)
    assert attention['camera_readout'].shape == (1, 4, 4, 4)
    assert attention['image_height_weights'].shape == (1, 2, 2, 2)
    torch.testing.assert_close(
        attention['lidar_readout'].sum(dim=-1),
        torch.ones(1, 4, 4),
        atol=1e-6,
        rtol=1e-5,
    )
