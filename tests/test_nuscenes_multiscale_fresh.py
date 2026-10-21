"""Fresh-fusion initialization, permanent freezing, schedule and artifact contracts."""

from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from mmengine.config import Config
from mmengine.optim import LinearLR, CosineAnnealingLR

from projects.myfusion.fusion.mbt_detector import MBTBEVFusionDetector
from projects.myfusion.hooks.fresh_fusion_training import (
    FreshFusionTrainingHook, VersionedNuScenesValidationHook)
from projects.myfusion.hooks.staged_fusion_training import StagedFusionTrainingHook

ROOT = Path(__file__).resolve().parents[1]


def config(name='nuscenes_multiscale_fresh.py'):
    return Config.fromfile(ROOT / 'configs' / name)


def test_full_config_preserves_data_and_uses_fresh_fixed_lidar_multiscale():
    cfg = config()
    original = config('nuscenes_camera_v2_no_mask.py')
    for key in ('train_dataloader', 'val_dataloader', 'class_names', 'point_cloud_range'):
        assert cfg[key] == original[key]
    assert cfg.model.type == 'NuScenesMultiScaleMBTDetectorV2'
    assert cfg.model.fusion_module.type == 'NuScenesMultiScaleMBTV2'
    assert cfg.model.fusion_module.num_bottleneck_tokens == 8
    assert cfg.model.fusion_module.embed_dim == 256
    assert cfg.model.img_backbone.init_cfg.checkpoint == 'torchvision://resnet50'
    assert cfg.model.camera_aux_loss_weight == .25
    assert cfg.model.fusion_module.lidar_spatial_drop_prob == 0
    assert cfg.resume is False and cfg.load_from == cfg.lidar_checkpoint
    assert 'pointpillars_nuscenes/' in cfg.load_from
    assert cfg.load_from.endswith('epoch_24.pth')
    assert cfg.train_cfg.type == 'IterBasedTrainLoop'
    assert 'max_epochs' not in cfg.train_cfg and 'by_epoch' not in cfg.train_cfg
    assert cfg.train_cfg.max_iters // cfg.optim_wrapper.accumulative_counts == 40000
    assert cfg.train_cfg.val_begin == cfg.train_cfg.val_interval == 10000
    assert cfg.custom_hooks[0].freeze_epochs > 0
    assert cfg.custom_hooks[2].type == 'FreshFusionTrainingHook'
    assert cfg.custom_hooks[3].type == 'VersionedNuScenesValidationHook'
    assert cfg.default_hooks.checkpoint.by_epoch is False
    assert cfg.param_scheduler[0].end == 2000
    assert cfg.param_scheduler[1].begin == 2000
    assert cfg.param_scheduler[1].end == cfg.train_cfg.max_iters
    assert cfg.param_scheduler[1].T_max == 158000
    assert cfg.param_scheduler[1].eta_min_ratio == .01
    assert cfg.train_cfg.val_interval % cfg.default_hooks.checkpoint.interval == 0


def test_lower_lr_changes_only_lr_scale_and_output_paths():
    first, second = config(), config('nuscenes_multiscale_fresh_lower_lr.py')
    assert second.optim_wrapper.optimizer.lr == 1e-4
    second.optim_wrapper.optimizer.lr = first.optim_wrapper.optimizer.lr
    for key in ('work_dir', 'val_evaluator', 'test_evaluator'):
        second[key] = first[key]
    assert second.to_dict() == first.to_dict()


def test_schedule_boundaries_and_component_lr_ratios():
    cfg = config()
    params = [nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(()))]
    optimizer = torch.optim.SGD([
        dict(params=[params[0]], lr=3e-4), dict(params=[params[1]], lr=6e-5)])
    schedulers = []
    for definition in cfg.param_scheduler:
        arguments = dict(definition)
        cls = {'LinearLR': LinearLR, 'CosineAnnealingLR': CosineAnnealingLR}[arguments.pop('type')]
        schedulers.append(cls(optimizer, **arguments))
    assert optimizer.param_groups[0]['lr'] == pytest.approx(3e-6)
    snapshots = {}
    for iteration in range(160000):
        if iteration % 4 == 3:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()
        if iteration in (1998, 1999, 2000, 80999, 159999):
            snapshots[iteration] = [group['lr'] for group in optimizer.param_groups]
    assert snapshots[1999][0] == pytest.approx(3e-4, rel=1e-6)
    assert 3e-6 < snapshots[80999][0] < 3e-4
    assert snapshots[159999][0] == pytest.approx(3e-6, rel=1e-4)
    for rates in snapshots.values():
        assert rates[1] / rates[0] == pytest.approx(.2, rel=1e-6)


@pytest.mark.parametrize('mode,transform', [
    ('black', 'BlackImage'), ('shuffled', 'ShuffleNuScenesImagePaths')])
def test_fresh_camera_ablations_preserve_architecture(mode, transform):
    original = config()
    cfg = config(f'nuscenes_multiscale_fresh_{mode}_eval.py')
    assert cfg.model == original.model
    assert [step for step in cfg.test_pipeline if step.type != transform] == original.test_pipeline
    assert cfg.test_dataloader.dataset.pipeline == cfg.test_pipeline


def fixture(tmp_path):
    model = nn.Module()
    for name in FreshFusionTrainingHook.module_names:
        setattr(model, name, nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3)))
    model._remap_pointpillars_state_dict = MBTBEVFusionDetector._remap_pointpillars_state_dict
    module = model.fusion_module = nn.Module()
    module.lidar_projection = nn.Linear(3, 3)
    module.pyramid_queries = nn.ModuleDict({str(i): deepcopy(module.lidar_projection) for i in (1, 2)})
    module.bev_residual = nn.Linear(3, 3)
    module.pyramid_residuals = nn.ModuleDict({str(i): nn.Linear(3, 3) for i in (1, 2)})
    for adapter in [module.bev_residual, *module.pyramid_residuals.values()]:
        nn.init.zeros_(adapter.weight)
        nn.init.zeros_(adapter.bias)
    module.lidar_spatial_drop_prob = 0
    source = {}
    for native, tensor in model.state_dict().items():
        for upstream, destination in MBTBEVFusionDetector._POINTPILLARS_PREFIX_MAP.items():
            if native.startswith(destination):
                source[upstream + native[len(destination):]] = tensor.clone()
    path = tmp_path / 'lidar.pth'
    torch.save(dict(state_dict=source), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    runner = SimpleNamespace(model=model, work_dir=str(tmp_path), iter=0, epoch=0,
        _has_loaded=True, cfg=SimpleNamespace(load_from=str(path)),
        optim_wrapper=SimpleNamespace(optimizer=optimizer, _accumulative_counts=4),
        logger=logging.getLogger('fresh-test'))
    return runner, path, digest


def test_guard_transfers_upstream_names_and_checks_full_frozen_state(tmp_path):
    runner, path, digest = fixture(tmp_path)
    hook = FreshFusionTrainingHook(str(path), digest)
    hook.before_train(runner)
    StagedFusionTrainingHook(1).before_train_epoch(runner)
    hook.before_train_epoch(runner)
    assert runner.model.fusion_module.lidar_projection.weight.requires_grad
    hook.after_train(runner)
    report = json.loads((tmp_path / 'fresh_fusion_integrity.json').read_text())
    assert report['exact_lidar_transfer'] and report['initial_residuals_zero']
    assert report['lidar_parameters_and_buffers_unchanged']


@pytest.mark.parametrize('mutation', ['wrong_hash', 'bn_buffer', 'nonzero_adapter', 'wrong_initializer'])
def test_guard_rejects_invalid_cold_start(tmp_path, mutation):
    runner, path, digest = fixture(tmp_path)
    if mutation == 'wrong_hash':
        digest = '0' * 64
    elif mutation == 'bn_buffer':
        runner.model.backbone[1].running_mean.add_(1)
    elif mutation == 'nonzero_adapter':
        with torch.no_grad():
            runner.model.fusion_module.bev_residual.weight.add_(1)
    else:
        runner.cfg.load_from = str(tmp_path / 'wrong.pth')
    with pytest.raises(RuntimeError):
        FreshFusionTrainingHook(str(path), digest).before_train(runner)


def test_resume_accepts_trained_fusion_but_not_changed_lidar(tmp_path):
    runner, path, digest = fixture(tmp_path)
    runner.iter = 8
    with torch.no_grad():
        runner.model.fusion_module.bev_residual.weight.add_(1)
    hook = FreshFusionTrainingHook(str(path), digest)
    hook.before_train(runner)
    assert hook.report['resumed']
    runner.model.neck[1].running_var.add_(1)
    with pytest.raises(RuntimeError, match='tensor changed'):
        hook._check(runner)
    hook.step_handle.remove()


def test_validation_predictions_get_distinct_directories(tmp_path):
    NuScenesMetric = type('NuScenesMetric', (), {})
    metric = NuScenesMetric()
    runner = SimpleNamespace(work_dir=str(tmp_path), iter=10000,
        val_loop=SimpleNamespace(evaluator=SimpleNamespace(metrics=[metric]),
                                 dataloader=SimpleNamespace(dataset=range(6019))))
    hook = VersionedNuScenesValidationHook()
    hook.before_val(runner)
    first = metric.jsonfile_prefix
    hook.after_val_epoch(runner, {'NDS': .45})
    assert json.loads((Path(first) / 'metrics.json').read_text())['frames'] == 6019
    with pytest.raises(FileExistsError):
        hook.before_val(runner)
    runner.iter = 20000
    hook.before_val(runner)
    assert metric.jsonfile_prefix != first
