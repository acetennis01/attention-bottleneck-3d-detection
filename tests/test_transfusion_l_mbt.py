"""Contracts for the separate pretrained TransFusion-L integration."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from mmengine.config import Config

from projects.myfusion.fusion.transfusion_l_mbt import TransFusionLMBTDetector
from projects.myfusion.hooks.transfusion_l_initialization import TransFusionLInitializationHook
from projects.myfusion.tools.transfusion_l_checkpoint import state_for_model

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return Config.fromfile(ROOT / 'configs' / name)


def test_pretrained_baseline_keeps_official_detector_not_pointpillars():
    cfg = config('transfusion_l_nuscenes_pretrained.py')
    original = Config.fromfile(ROOT.parent / 'BEVFusion/configs/'
        'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py')
    assert cfg.model == original.model
    assert cfg.point_cloud_range == [-54., -54., -5., 54., 54., 3.]
    assert cfg.voxel_size == [.075, .075, .2]
    assert cfg.class_names == original.class_names
    assert cfg.model.bbox_head.type == 'TransFusionHead'
    assert cfg.model.bbox_head.in_channels == 512
    assert cfg.train_cfg is None and cfg.train_dataloader is None
    assert cfg.test_pipeline[0].use_dim == 5
    assert cfg.test_pipeline[1].sweeps_num == 9
    assert cfg.test_pipeline[1].test_mode is True
    assert cfg.test_dataloader.dataset.get('indices') is None


def test_mbt_keeps_complete_pretrained_lidar_and_head_contract():
    base = config('transfusion_l_nuscenes_pretrained.py')
    cfg = config('transfusion_l_mbt_nuscenes.py')
    for key in ('pts_voxel_encoder', 'pts_middle_encoder', 'pts_backbone',
                'pts_neck', 'bbox_head'):
        assert cfg.model[key] == base.model[key]
    assert cfg.model.data_preprocessor.voxelize_cfg == base.model.data_preprocessor.voxelize_cfg
    assert cfg.model.type == 'TransFusionLMBTDetector'
    assert cfg.model.freeze_lidar is True
    fusion = cfg.model.fusion_module
    assert fusion.point_cloud_range == cfg.point_cloud_range
    assert fusion.in_lidar_channels == 512
    assert fusion.num_bottleneck_tokens == 8 and fusion.num_layers == 4
    assert fusion.lidar_spatial_drop_prob == 0
    assert fusion.use_local_camera_residual is False
    assert cfg.load_from is None  # Mandatory audited initialization hook instead.
    assert cfg.custom_hooks[0].type == 'TransFusionLInitializationHook'
    assert cfg.optim_wrapper.accumulative_counts == 4
    assert cfg.train_cfg.max_iters // 4 == 40000
    assert cfg.test_dataloader.dataset.get('indices') is None
    assert [step for step in cfg.test_pipeline if step.type.startswith('LoadPoints')
            or step.type == 'PointsRangeFilter'] == base.test_pipeline[:-1]
    assert all(step.type not in ('ObjectSample', 'GlobalRotScaleTrans',
        'RandomFlip3D') for step in cfg.train_pipeline)


def test_axis_conversion_is_geometric_not_just_square_shape():
    # Detector dimension 2 is X, dimension 3 is Y. Use a rectangular map and
    # distinct coordinate values so a transposition error cannot pass unnoticed.
    original = torch.arange(3)[:, None] * 100 + torch.arange(7)[None, :]
    original = original[None, None].float()
    sampler = TransFusionLMBTDetector.to_sampler_bev(original)
    assert sampler.shape == (1, 1, 7, 3)
    assert sampler[0, 0, 5, 2] == 205
    assert torch.equal(TransFusionLMBTDetector.to_detector_bev(sampler), original)


def test_explicit_sparse_weight_permutation_is_reversible():
    model = nn.Module()
    model.pts_middle_encoder = nn.Module()
    model.pts_middle_encoder.conv = nn.Module()
    model.pts_middle_encoder.conv.weight = nn.Parameter(torch.zeros(2, 3, 4, 5, 7))
    raw = torch.arange(7 * 2 * 3 * 4 * 5).reshape(7, 2, 3, 4, 5).float()
    state = {'pts_middle_encoder.conv.weight': raw}
    converted, changed = state_for_model(model, state)
    assert changed == ['pts_middle_encoder.conv.weight']
    assert torch.equal(converted[changed[0]].permute(4, 0, 1, 2, 3), raw)
    assert state['pts_middle_encoder.conv.weight'] is raw
    again, changed = state_for_model(model, converted)
    assert changed == []
    assert torch.equal(again['pts_middle_encoder.conv.weight'], converted['pts_middle_encoder.conv.weight'])


def test_unrelated_checkpoint_shape_mismatch_is_not_silently_adapted():
    model = nn.Linear(3, 2)
    with pytest.raises(RuntimeError, match='Unsupported checkpoint shape mismatch'):
        state_for_model(model, {'weight': torch.zeros(4, 3)})


def small_frozen_detector():
    model = TransFusionLMBTDetector.__new__(TransFusionLMBTDetector)
    nn.Module.__init__(model)
    model.freeze_lidar = True
    for name in model.lidar_module_names:
        setattr(model, name, nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3)))
    model.img_backbone = nn.Linear(3, 3)
    model._set_lidar_frozen()
    return model


def test_train_mode_does_not_unfreeze_lidar_or_batchnorm():
    model = small_frozen_detector()
    for mode in (True, False, True):
        model.train(mode)
        assert model.img_backbone.training == mode
        for name in model.lidar_module_names:
            module = getattr(model, name)
            assert not module.training
            assert all(not parameter.requires_grad for parameter in module.parameters())


@pytest.mark.parametrize('status', ['running', 'failed', 'mismatch'])
def test_training_gate_rejects_unverified_baselines(tmp_path, status):
    path = tmp_path / 'baseline.json'
    path.write_text(json.dumps(dict(status=status, validation_frames=6019)))
    hook = TransFusionLInitializationHook('unused.pth', str(path))
    with pytest.raises(RuntimeError, match='Reproduce the full LiDAR baseline'):
        hook.before_train(SimpleNamespace())


def test_integrity_guard_detects_frozen_bn_buffer_drift(tmp_path):
    model = small_frozen_detector()
    reference = {key: value.clone() for key, value in model.state_dict().items()
                 if key.startswith(('pts_', 'bbox_head.'))}
    hook = TransFusionLInitializationHook('unused.pth', 'baseline.json')
    hook.reference = reference
    runner = SimpleNamespace(model=model, iter=0, work_dir=str(tmp_path))
    hook._verify(runner)
    model.pts_backbone[1].running_mean[0] += 1
    with pytest.raises(RuntimeError, match='Pretrained LiDAR/head state changed'):
        hook._verify(runner)
