"""Focused tests for the separate six-camera nuScenes implementation."""

import torch

from projects.myfusion.fusion.mbt_bev_nuscenes import (
    NuScenesSymmetricMBTBEVFusion,
)
from projects.myfusion.fusion.mbt_detector_nuscenes import (
    NuScenesMBTBEVFusionDetector,
)


def test_six_view_fusion_shapes_and_validity():
    module = NuScenesSymmetricMBTBEVFusion(
        in_img_channels=8,
        in_lidar_channels=8,
        embed_dim=8,
        num_heads=2,
        token_grid_size=(2, 2),
        num_bottleneck_tokens=2,
        num_layers=2,
        point_cloud_range=(-1, -1, 0, 1, 1, 2),
        height_anchors=(1.0, ),
        use_local_camera_residual=False,
        camera_aux_num_classes=10,
    )
    images = torch.randn(1, 6, 8, 4, 4)
    lidar = torch.randn(1, 8, 4, 4)
    projection = torch.tensor([
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]).reshape(1, 1, 4, 4).expand(1, 6, 4, 4)
    shapes = torch.tensor([[[4.0, 4.0]]]).expand(1, 6, 2)

    output = module(
        images,
        lidar,
        lidar2img=projection,
        image_shapes=shapes,
        padded_image_shape=(4, 4),
        return_attention=True,
    )

    assert output['bev_features'].shape == lidar.shape
    assert output['camera_aux_logits'].shape == (1, 4, 10)
    assert output['image_valid_mask'].shape == (1, 4, 1)
    assert output['image_valid_mask'].all()
    assert output['attention']['image_view_height_weights'].shape == (
        1, 6, 1, 2, 2)


def test_lidar_to_image_composition_supports_three_by_three_intrinsics():
    intrinsics = torch.tensor([
        [2.0, 0.0, 1.0],
        [0.0, 3.0, 1.5],
        [0.0, 0.0, 1.0],
    ]).reshape(1, 3, 3)
    extrinsics = torch.eye(4).reshape(1, 4, 4)
    result = NuScenesMBTBEVFusionDetector._compose_lidar2img(
        intrinsics, extrinsics)

    assert result.shape == (1, 4, 4)
    assert torch.equal(result[0, :3, :3], intrinsics[0])


def test_pointpillars_checkpoint_prefixes_are_remapped_for_warm_start():
    state_dict = {
        'pts_voxel_encoder.layer.weight': torch.ones(1),
        'pts_middle_encoder.layer.weight': torch.ones(1),
        'pts_backbone.layer.weight': torch.ones(1),
        'pts_neck.layer.weight': torch.ones(1),
        'pts_bbox_head.layer.weight': torch.ones(1),
        'img_backbone.layer.weight': torch.ones(1),
    }

    NuScenesMBTBEVFusionDetector._remap_pointpillars_state_dict(state_dict)

    assert 'voxel_encoder.layer.weight' in state_dict
    assert 'middle_encoder.layer.weight' in state_dict
    assert 'backbone.layer.weight' in state_dict
    assert 'neck.layer.weight' in state_dict
    assert 'bbox_head.layer.weight' in state_dict
    assert 'img_backbone.layer.weight' in state_dict
    assert not any(key.startswith('pts_') for key in state_dict)


def test_pointpillars_checkpoint_loads_into_detector_modules():
    detector = NuScenesMBTBEVFusionDetector.__new__(
        NuScenesMBTBEVFusionDetector)
    torch.nn.Module.__init__(detector)
    module_names = (
        'voxel_encoder',
        'middle_encoder',
        'backbone',
        'neck',
        'bbox_head',
    )
    for name in module_names:
        setattr(detector, name, torch.nn.Linear(1, 1, bias=False))

    checkpoint = {
        f'pts_{name}.weight': torch.full((1, 1), index + 1.0)
        for index, name in enumerate(module_names)
    }
    incompatible = detector.load_state_dict(checkpoint, strict=False)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for index, name in enumerate(module_names):
        expected = torch.full((1, 1), index + 1.0)
        assert torch.equal(getattr(detector, name).weight, expected)
