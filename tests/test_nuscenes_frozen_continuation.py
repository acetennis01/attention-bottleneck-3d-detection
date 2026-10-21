"""Config contract and frozen-state guards for the bounded continuation."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import logging

import pytest
import torch
from torch import nn
from mmengine.config import Config

from projects.myfusion.hooks.frozen_lidar_integrity import FrozenLiDARIntegrityHook
from projects.myfusion.hooks.staged_fusion_training import StagedFusionTrainingHook

ROOT = Path(__file__).resolve().parents[1]


def test_continuation_preserves_architecture_and_optimization_contract():
    old = Config.fromfile(ROOT / 'configs/nuscenes_camera_v2_no_mask.py')
    cfg = Config.fromfile(ROOT / 'configs/nuscenes_camera_v2_frozen_lidar_pilot.py')
    assert cfg.model == old.model
    assert cfg.optim_wrapper == old.optim_wrapper
    assert cfg.param_scheduler == old.param_scheduler
    assert cfg.train_dataloader == old.train_dataloader
    assert cfg.val_dataloader == old.val_dataloader
    assert cfg.class_names == old.class_names
    assert cfg.randomness == old.randomness
    assert cfg.point_cloud_range == old.point_cloud_range
    assert cfg.load_from.endswith('NuScenes_NDS_epoch_5.pth')
    assert 'nomask_seed0/' in cfg.load_from
    assert cfg.resume is False
    assert cfg.train_cfg.max_epochs == 2
    assert cfg.train_cfg.val_begin == cfg.train_cfg.val_interval == 1
    assert cfg.custom_hooks[0].freeze_epochs >= 24
    assert cfg.custom_hooks[1].max_drop_prob == 0
    assert cfg.custom_hooks[-1].reference_checkpoint == cfg.load_from
    assert cfg.work_dir != old.work_dir
    assert cfg.val_evaluator.jsonfile_prefix != old.val_evaluator.jsonfile_prefix


def fixture_runner(tmp_path):
    model = nn.Module()
    model.backbone = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    model.bbox_head = nn.Linear(3, 2)
    model.img_backbone = nn.Linear(3, 3)
    path = tmp_path / 'reference.pth'
    torch.save(dict(state_dict=deepcopy(model.state_dict())), path)
    runner = SimpleNamespace(model=model, work_dir=str(tmp_path), epoch=0, iter=0,
                             logger=logging.getLogger('frozen-integrity-test'))
    return runner, path


def test_guard_checks_weights_bn_buffers_and_trainable_camera(tmp_path):
    runner, path = fixture_runner(tmp_path)
    guard = FrozenLiDARIntegrityHook(str(path))
    staged = StagedFusionTrainingHook(freeze_epochs=24)
    guard.before_train(runner)
    for epoch in (0, 1, 5, 23):
        runner.epoch = epoch
        runner.model.train()
        staged.before_train_epoch(runner)
        guard.before_train_epoch(runner)
        assert runner.model.img_backbone.training
        assert all(p.requires_grad for p in runner.model.img_backbone.parameters())
        guard.after_train_epoch(runner)
    assert (tmp_path / 'frozen_lidar_integrity.json').is_file()


@pytest.mark.parametrize('mutation', ['weight', 'running_mean', 'running_var', 'num_batches_tracked'])
def test_guard_rejects_changed_parameter_or_bn_buffer(tmp_path, mutation):
    runner, path = fixture_runner(tmp_path)
    guard = FrozenLiDARIntegrityHook(str(path))
    guard.before_train(runner)
    StagedFusionTrainingHook(24).before_train_epoch(runner)
    with torch.no_grad():
        if mutation == 'weight':
            runner.model.backbone[0].weight.add_(1)
        else:
            getattr(runner.model.backbone[1], mutation).add_(1)
    with pytest.raises(RuntimeError, match='state changed'):
        guard.after_train_epoch(runner)


def test_guard_rejects_accidental_unfreeze_and_training_mode(tmp_path):
    runner, path = fixture_runner(tmp_path)
    guard = FrozenLiDARIntegrityHook(str(path))
    guard.before_train(runner)
    staged = StagedFusionTrainingHook(24)
    staged.before_train_epoch(runner)
    runner.model.backbone[0].weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match='enabled gradients'):
        guard.before_train_epoch(runner)
    staged.before_train_epoch(runner)
    runner.model.backbone[1].train()
    with pytest.raises(RuntimeError, match='training mode'):
        guard.before_train_epoch(runner)
