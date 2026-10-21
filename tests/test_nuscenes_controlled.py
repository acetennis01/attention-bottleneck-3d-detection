"""Controlled update budgets and added-scale camera paths."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from mmengine.config import Config
from torch import nn

from projects.myfusion.fusion.mbt_nuscenes_multiscale import (
    NuScenesMultiScaleMBTV2 as Multi,
    NuScenesMultiScaleMBTDetectorV2 as Detector)
from projects.myfusion.fusion.mbt_bev_nuscenes_camera_v2 import (
    NuScenesCameraFocusedMBTBEVFusionV2 as Single)
from projects.myfusion.fusion.mbt_detector_nuscenes_camera_v2 import (
    NuScenesCameraFocusedMBTBEVFusionDetectorV2)

ROOT = Path(__file__).resolve().parents[1]


def configuration(arm):
    return Config.fromfile(ROOT / f'configs/nuscenes_controlled_{arm}.py')


def test_equal_exposure_and_one_factor_controls():
    frozen = configuration('frozen')
    for arm in ('joint', 'lower_lr', 'lower_aux', 'multiscale'):
        cfg = configuration(arm)
        for name in ('train_cfg', 'train_dataloader', 'val_dataloader',
                     'randomness', 'param_scheduler', 'class_names',
                     'point_cloud_range', 'default_hooks'):
            assert cfg[name] == frozen[name], (arm, name)
        assert cfg.resume is False
        assert cfg.optim_wrapper.accumulative_counts == 4
        assert cfg.train_cfg.max_iters == 4000
        assert cfg.train_cfg.val_interval == cfg.train_cfg.val_begin == 4000
        assert cfg.train_cfg.type == 'IterBasedTrainLoop'
        assert 'max_epochs' not in cfg.train_cfg and 'by_epoch' not in cfg.train_cfg
        assert cfg.work_dir != frozen.work_dir
        assert cfg.custom_hooks[-1].reference_checkpoint == frozen.reference_checkpoint
    joint = configuration('joint')
    assert joint.model == frozen.model and joint.optim_wrapper == frozen.optim_wrapper
    assert joint.custom_hooks[0].freeze_epochs == 0
    assert joint.custom_hooks[-1].frozen_lidar is False
    low_lr = configuration('lower_lr')
    assert low_lr.model == frozen.model and low_lr.custom_hooks == frozen.custom_hooks
    low_lr.optim_wrapper.optimizer.lr = 3e-4
    assert low_lr.optim_wrapper == frozen.optim_wrapper
    low_aux = configuration('lower_aux')
    assert low_aux.optim_wrapper == frozen.optim_wrapper
    assert low_aux.model.camera_aux_loss_weight == 0.05
    low_aux.model.camera_aux_loss_weight = .25
    assert low_aux.model == frozen.model
    multi = configuration('multiscale')
    assert multi.optim_wrapper == frozen.optim_wrapper
    multi.model.type = frozen.model.type
    multi.model.fusion_module.type = frozen.model.fusion_module.type
    assert multi.model == frozen.model


def build(cls):
    return cls(in_img_channels=8, in_lidar_channels=12, embed_dim=16,
               num_heads=4, num_layers=2, token_grid_size=(2, 2),
               num_bottleneck_tokens=8, point_cloud_range=(-1, -1, 0, 1, 1, 2),
               height_anchors=(.5, 1.), camera_aux_num_classes=10,
               proj_dropout=0., lidar_spatial_drop_prob=0.)


def inputs():
    image = torch.randn(1, 6, 8, 4, 4)
    pyramid = (torch.randn(1, 12, 7, 9), torch.randn(1, 12, 4, 5),
               torch.randn(1, 12, 2, 3))
    projection = torch.tensor([[1., 0., 2., 0.], [0., 1., 2., 0.],
                               [0., 0., 1., 0.], [0., 0., 0., 1.]])
    return image, pyramid, dict(
        lidar2img=projection.reshape(1, 1, 4, 4).repeat(1, 6, 1, 1),
        image_shapes=torch.full((1, 6, 2), 4.), padded_image_shape=(4, 4))


def test_warm_start_preserves_all_old_features_and_strict_reload():
    old, multi = build(Single).eval(), build(Multi).eval()
    nn.init.normal_(old.bev_residual.weight, std=.02)
    missing, unexpected = multi.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and len(missing) == 8
    assert all(key.startswith(('pyramid_queries.', 'pyramid_residuals.')) for key in missing)
    image, pyramid, meta = inputs()
    with torch.no_grad():
        before, after = old(image, pyramid[0], **meta), multi(image, pyramid[0], **meta)
        for key in ('bev_features', 'bottleneck_tokens', 'camera_aux_logits'):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        for level in (1, 2):
            output = multi.fuse_pyramid_level(pyramid[level], after['bottleneck_tokens'], level)
            torch.testing.assert_close(output, pyramid[level], rtol=0, atol=0)
    build(Multi).load_state_dict(multi.state_dict(), strict=True)


@pytest.mark.parametrize('level', [1, 2])
def test_added_scale_has_camera_path_only_through_shared_bottlenecks(level):
    module = build(Multi).eval()
    # At initialization only the zero adapter learns. After it opens, detection
    # gradients must reach the shared B tokens and camera, without auxiliary loss.
    nn.init.normal_(module.pyramid_residuals[str(level)].weight, std=.02)
    image, pyramid, meta = inputs()
    image.requires_grad_()
    output = module(image, pyramid[0], **meta)
    module.fuse_pyramid_level(
        pyramid[level], output['bottleneck_tokens'], level).square().mean().backward()
    assert (image.grad.abs().flatten(2).sum(2) > 0).all()
    assert module.bottleneck_tokens.grad.abs().sum() > 0
    assert module.camera_aux_head.weight.grad is None
    module.zero_grad(set_to_none=True)
    image.grad = None
    output = module(image, pyramid[0], **meta)
    module.fuse_pyramid_level(pyramid[level], output['bottleneck_tokens'].detach(), level).sum().backward()
    assert image.grad is None


def test_detector_replaces_all_maps_without_changing_level_zero():
    detector = Detector.__new__(Detector)
    nn.Module.__init__(detector)
    detector.fusion_level = 0
    detector.fusion_module = build(Multi).eval()
    image, pyramid, meta = inputs()
    output = detector.fusion_module(image, pyramid[0], **meta)
    features = (output['bev_features'], *pyramid[1:])
    with patch.object(NuScenesCameraFocusedMBTBEVFusionDetectorV2, 'extract_feat',
                      return_value=(features, output)):
        result, same_output = detector.extract_feat({}, return_fusion_output=True)
    assert same_output is output and result[0] is features[0]
    for before, after in zip(features, result):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_hard_mask_must_not_be_silently_enabled():
    with pytest.raises(ValueError, match='masking disabled'):
        Multi(in_img_channels=8, in_lidar_channels=12, embed_dim=16,
              num_heads=4, point_cloud_range=(-1, -1, 0, 1, 1, 2),
              lidar_spatial_drop_prob=.1)
