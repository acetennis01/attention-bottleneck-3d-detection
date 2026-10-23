"""Strict camera pretraining and unchanged MBT/LiDAR contracts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config
from mmdet3d.registry import MODELS

from projects.myfusion.fusion.nuimages_swin import (
    camera_state_for_backbone, NUIMAGES_SWIN_SHA256)

ROOT = Path(__file__).resolve().parents[1]


def test_camera_change_preserves_detector_fusion_and_data_recipe():
    first = Config.fromfile(ROOT / 'configs/transfusion_l_mbt_nuscenes.py')
    cfg = Config.fromfile(ROOT / 'configs/transfusion_l_mbt_swint_nuscenes.py')
    for name in ('pts_voxel_encoder', 'pts_middle_encoder', 'pts_backbone',
                 'pts_neck', 'bbox_head', 'fusion_module'):
        assert cfg.model[name] == first.model[name]
    for name in ('train_dataloader', 'val_dataloader', 'test_dataloader',
                 'train_pipeline', 'test_pipeline', 'point_cloud_range',
                 'class_names', 'param_scheduler'):
        assert cfg[name] == first[name]
    optimizer = deepcopy(cfg.optim_wrapper)
    assert optimizer.clip_grad.pop('error_if_nonfinite') is True
    assert optimizer == first.optim_wrapper
    assert cfg.train_cfg.max_iters == first.train_cfg.max_iters
    assert cfg.model.img_backbone.type == 'NuImagesSwinTransformer'
    assert cfg.model.img_backbone.convert_weights is False
    assert cfg.model.img_backbone.mlp_ratio == 4
    assert isinstance(cfg.model.img_backbone.mlp_ratio, int)
    assert cfg.model.img_backbone.with_cp is True
    assert cfg.model.img_backbone.frozen_stages == 1
    assert cfg.model.img_neck.in_channels == [192, 384, 768]
    assert cfg.model.img_neck.out_channels == 256
    assert cfg.model.data_preprocessor.bgr_to_rgb is True
    assert cfg.camera_checkpoint_sha256 == NUIMAGES_SWIN_SHA256


def test_early_validation_schedule_uses_installed_mmengine_interval_logic():
    from mmengine.runner.loops import IterBasedTrainLoop
    from mmengine.runner.utils import calc_dynamic_intervals
    cfg = Config.fromfile(ROOT / 'configs/transfusion_l_mbt_swint_nuscenes.py')
    loop = SimpleNamespace()
    loop.dynamic_milestones, loop.dynamic_intervals = calc_dynamic_intervals(
        cfg.train_cfg.val_interval, cfg.train_cfg.dynamic_intervals)
    scheduled = []
    for iteration in range(1, 20001):
        loop._iter = iteration
        IterBasedTrainLoop._decide_current_val_interval(loop)
        if iteration >= cfg.train_cfg.val_begin and iteration % loop.val_interval == 0:
            scheduled.append(iteration)
    assert scheduled == [2000, 10000, 20000]


def native_fixture():
    state = {'patch_embed.projection.weight': torch.ones(2, 3, 4, 4),
             'norm1.weight': torch.ones(2), 'norm1.bias': torch.zeros(2)}
    backbone = SimpleNamespace(out_indices=(1, 2, 3), state_dict=lambda: state)
    source = deepcopy(state)
    source.update({'norm0.weight': torch.ones(1), 'norm0.bias': torch.zeros(1)})
    return backbone, state, source


def test_only_unused_stage_output_norm_is_excluded():
    backbone, expected, source = native_fixture()
    state, unused = camera_state_for_backbone(backbone, source)
    assert set(state) == set(expected)
    assert unused == ['norm0.bias', 'norm0.weight']
    assert all(torch.equal(value, source[key]) for key, value in state.items())


@pytest.mark.parametrize('key', ['head.weight', 'backbone.patch_embed.projection.weight',
                               'stages.0.foreign.weight'])
def test_unexpected_weights_are_never_silently_ignored(key):
    backbone, _, source = native_fixture()
    source[key] = torch.zeros(1)
    with pytest.raises(RuntimeError, match='Unexpected camera checkpoint tensors'):
        camera_state_for_backbone(backbone, source)


def test_missing_camera_parameters_are_fatal():
    backbone, _, source = native_fixture()
    del source['patch_embed.projection.weight']
    with pytest.raises(RuntimeError, match='Missing pretrained camera tensors'):
        camera_state_for_backbone(backbone, source)


def test_no_camera_tensor_resizing_or_interpolation():
    backbone, _, source = native_fixture()
    source['patch_embed.projection.weight'] = torch.zeros(3, 3, 4, 4)
    with pytest.raises(RuntimeError, match='Camera tensor shape mismatch'):
        camera_state_for_backbone(backbone, source)


def test_downloaded_checkpoint_matches_reference_mmdet_swin_forward():
    cfg = Config.fromfile(ROOT / 'configs/transfusion_l_mbt_swint_nuscenes.py')
    path = Path(cfg.camera_checkpoint)
    if not path.is_file():
        pytest.skip('Download the official camera checkpoint for this integration test')
    definition = deepcopy(cfg.model.img_backbone)
    backbone = MODELS.build(definition)
    backbone.eval()  # MMDetection's Swin.train() does not return self.
    backbone.init_weights()
    verification = backbone.pretrained_verification
    assert verification['loaded_tensors'] == 187
    assert verification['source_tensor_count'] == 189
    assert verification['unused_output_norm_tensors'] == ['norm0.bias', 'norm0.weight']
    definition.type = 'mmdet.SwinTransformer'
    definition.init_cfg = None
    reference = MODELS.build(definition)
    reference.eval()
    source = torch.load(path, map_location='cpu')['state_dict']
    state = {key: source[key] for key in reference.state_dict()}
    reference.load_state_dict(state, strict=True)
    with torch.no_grad():
        images = torch.randn(1, 3, 64, 96)
        actual, expected = backbone(images), reference(images)
    assert [tensor.shape[1] for tensor in actual] == [192, 384, 768]
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    # Runner may call init_weights again; never reset already loaded/trained
    # parameters on that second call.
    with torch.no_grad():
        backbone.patch_embed.projection.weight[0, 0, 0, 0] += 1
    changed = backbone.patch_embed.projection.weight.clone()
    backbone.init_weights()
    assert torch.equal(backbone.patch_embed.projection.weight, changed)
