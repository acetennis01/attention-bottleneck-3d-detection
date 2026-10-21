"""Verified LiDAR transfer, permanent frozen-state checks and saved validation."""

import hashlib
import json
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class FreshFusionTrainingHook(Hook):
    """Guard a cold-fusion run and exact optimizer-state resumes of that run.

    Unlike the continuation guard, this accepts a LiDAR-only checkpoint with
    upstream pts_* names. All LiDAR weights and BN buffers must match it forever.
    Fresh starts require zero residuals and no optimizer state; resumes do not.
    """

    priority = 'LOW'
    module_names = ('voxel_encoder', 'middle_encoder', 'backbone', 'neck', 'bbox_head')

    def __init__(self, reference_checkpoint, reference_sha256, interval=1000):
        if interval < 1:
            raise ValueError('interval must be positive')
        self.reference_checkpoint = reference_checkpoint
        self.reference_sha256 = reference_sha256
        self.interval = interval
        self.steps = 0

    @staticmethod
    def _model(runner):
        return runner.model.module if is_model_wrapper(runner.model) else runner.model

    def _write(self, runner):
        path = Path(runner.work_dir) / 'fresh_fusion_integrity.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.report, indent=2) + '\n')
        temporary.replace(path)

    def _check(self, runner, modes=False):
        model = self._model(runner)
        actual = {f'{name}.{key}': tensor for name in self.module_names
                  for key, tensor in getattr(model, name).state_dict().items()}
        if set(actual) != set(self.reference):
            raise RuntimeError('LiDAR/head reference keys differ from the detector')
        for key, value in actual.items():
            if not torch.equal(value.detach().cpu(), self.reference[key]):
                raise RuntimeError(f'Fixed LiDAR/head tensor changed: {key}')
        if modes:
            for name in self.module_names:
                module = getattr(model, name)
                if any(child.training for child in module.modules()):
                    raise RuntimeError(f'Fixed module is in training mode: {name}')
                if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                    raise RuntimeError(f'Fixed module has gradients: {name}')

    def before_train(self, runner):
        path = Path(self.reference_checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.reference_sha256:
            raise RuntimeError('LiDAR initializer checksum differs from the configured reference')
        model = self._model(runner)
        source = dict(torch.load(path, map_location='cpu')['state_dict'])
        model._remap_pointpillars_state_dict(source)
        prefixes = tuple(name + '.' for name in self.module_names)
        if not source or any(not key.startswith(prefixes) for key in source):
            raise RuntimeError('Fresh fusion requires a LiDAR-only reference, not MBT weights')
        self.reference = source
        self._check(runner)
        resumed = bool(getattr(runner, '_has_loaded', False) and runner.iter > 0)
        accumulation = runner.optim_wrapper._accumulative_counts
        if runner.iter % accumulation:
            raise RuntimeError('Resume checkpoint splits a gradient-accumulation window')
        if not resumed:
            if runner.optim_wrapper.optimizer.state:
                raise RuntimeError('Fresh fusion must start with an empty optimizer state')
            if Path(runner.cfg.load_from).resolve() != path.resolve():
                raise RuntimeError('Fresh start must load the configured LiDAR-only reference')
            module = model.fusion_module
            adapters = [module.bev_residual, *module.pyramid_residuals.values()]
            if any(torch.count_nonzero(value.detach()) for adapter in adapters
                   for value in adapter.parameters()):
                raise RuntimeError('Fresh fusion output adapters must all be zero')
            # During model construction these copies precede the checkpoint load.
            # For a cold fusion start both copies must retain the same fresh query.
            for query in module.pyramid_queries.values():
                for key, value in query.state_dict().items():
                    if not torch.equal(value, module.lidar_projection.state_dict()[key]):
                        raise RuntimeError('Fresh pyramid query initialization diverged')
        self.initial_iter = int(runner.iter)
        self.steps = 0
        self.step_handle = runner.optim_wrapper.optimizer.register_step_post_hook(
            self._after_step)
        self.report = dict(
            status='training', reference_checkpoint=str(path.resolve()),
            reference_sha256=digest, lidar_tensors=len(source),
            exact_lidar_transfer=True, resumed=resumed,
            fresh_optimizer=not resumed, start_microbatch=self.initial_iter,
            accumulation=accumulation, optimizer_updates_this_session=0,
            camera_initialization='ImageNet ResNet-50; existing state on resume',
            fusion_initialization='new random MBT; existing state on resume',
            hard_mask_disabled=model.fusion_module.lidar_spatial_drop_prob == 0,
            initial_residuals_zero=not resumed)
        if not self.report['hard_mask_disabled']:
            raise RuntimeError('Fresh multi-scale recipe must not mask LiDAR')
        self._write(runner)

    def _after_step(self, optimizer, args, kwargs):
        self.steps += 1

    def before_train_epoch(self, runner):
        self._check(runner, modes=True)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not torch.isfinite(outputs['loss']).all():
            raise RuntimeError('Non-finite fresh-fusion training loss')
        if (runner.iter + 1) % self.interval == 0:
            self._check(runner, modes=True)
            self.report.update(completed_microbatches=runner.iter + 1,
                               optimizer_updates_this_session=self.steps)
            self._write(runner)

    def after_train(self, runner):
        self._check(runner, modes=True)
        expected = (runner.iter - self.initial_iter) // self.report['accumulation']
        if self.steps != expected:
            raise RuntimeError(f'Expected {expected} optimizer updates, got {self.steps}')
        self.report.update(status='complete', completed_microbatches=runner.iter,
                           optimizer_updates_this_session=self.steps,
                           total_optimizer_updates=runner.iter // self.report['accumulation'],
                           lidar_parameters_and_buffers_unchanged=True)
        self.step_handle.remove()
        self._write(runner)


@HOOKS.register_module()
class VersionedNuScenesValidationHook(Hook):
    """Do not overwrite earlier validation predictions or metrics."""

    priority = 'HIGH'

    def before_val(self, runner):
        prefix = Path(runner.work_dir) / 'validation' / f'iter_{runner.iter:06d}'
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if prefix.exists() and any(prefix.iterdir()):
            raise FileExistsError(f'Validation artifacts already exist: {prefix}')
        metrics = runner.val_loop.evaluator.metrics
        if len(metrics) != 1 or metrics[0].__class__.__name__ != 'NuScenesMetric':
            raise RuntimeError('Expected one NuScenesMetric evaluator')
        metrics[0].jsonfile_prefix = str(prefix)

    def after_val_epoch(self, runner, metrics=None):
        root = Path(runner.work_dir) / 'validation' / f'iter_{runner.iter:06d}'
        root.mkdir(parents=True, exist_ok=True)
        report = dict(iteration=int(runner.iter),
                      frames=len(runner.val_loop.dataloader.dataset),
                      metrics={key: float(value) for key, value in (metrics or {}).items()})
        (root / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
