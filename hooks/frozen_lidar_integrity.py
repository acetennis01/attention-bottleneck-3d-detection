"""Fail-fast checks for model-only continuations with a permanently fixed LiDAR path."""

from pathlib import Path
import hashlib
import json

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class FrozenLiDARIntegrityHook(Hook):
    """Check all LiDAR/head parameters AND buffers against a reference model.

    Runs after the staged-freezing hook. Full tensor comparisons occur only
    at startup and epoch boundaries. Gradient/mode assertions are sampled.
    Works on resume too: only the permanently frozen branch is compared.
    """

    priority = 'LOW'
    module_names = ('voxel_encoder', 'middle_encoder', 'backbone', 'neck',
                    'bbox_head')

    def __init__(self, reference_checkpoint: str, interval: int = 500):
        if interval < 1:
            raise ValueError('interval must be positive')
        self.reference_checkpoint = reference_checkpoint
        self.interval = interval
        self.reference = None
        self.report = {}

    @staticmethod
    def _model(runner):
        return (runner.model.module if is_model_wrapper(runner.model)
                else runner.model)

    def _modules(self, runner):
        model = self._model(runner)
        for name in self.module_names:
            module = getattr(model, name, None)
            if module is not None:
                yield name, module

    def _write(self, runner):
        path = Path(runner.work_dir) / 'frozen_lidar_integrity.json'
        path.write_text(json.dumps(self.report, indent=2) + '\n')

    def _check_state(self, runner):
        actual = {f'{name}.{key}': value for name, module in
                  self._modules(runner) for key, value in module.state_dict().items()}
        if set(actual) != set(self.reference):
            raise RuntimeError('Frozen LiDAR/head state keys differ from reference')
        for key, value in actual.items():
            if not torch.equal(value.detach().cpu(), self.reference[key]):
                raise RuntimeError(f'Frozen LiDAR/head state changed: {key}')

    def _check_modes_and_gradients(self, runner):
        for name, module in self._modules(runner):
            if any(child.training for child in module.modules()):
                raise RuntimeError(f'Frozen module is in training mode: {name}')
            for parameter in module.parameters():
                if parameter.requires_grad or parameter.grad is not None:
                    raise RuntimeError(f'Frozen module has enabled gradients: {name}')

    def before_train(self, runner):
        path = Path(self.reference_checkpoint)
        checkpoint = torch.load(path, map_location='cpu')
        prefixes = tuple(name + '.' for name in self.module_names)
        self.reference = {key: value for key, value in
                          checkpoint['state_dict'].items() if key.startswith(prefixes)}
        if not self.reference:
            raise RuntimeError('No native MBT LiDAR/head tensors in reference')
        self._check_state(runner)
        self.report = dict(reference_checkpoint=str(path.resolve()),
                           reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           tensors=len(self.reference), startup_exact=True,
                           completed_epochs=[])
        self._write(runner)
        runner.logger.info('Frozen LiDAR integrity: exact reference match at startup')

    def before_train_epoch(self, runner):
        self._check_modes_and_gradients(runner)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if self.every_n_train_iters(runner, self.interval):
            self._check_modes_and_gradients(runner)

    def after_train_epoch(self, runner):
        self._check_modes_and_gradients(runner)
        self._check_state(runner)
        epoch = int(runner.epoch) + 1
        self.report['completed_epochs'].append(epoch)
        self._write(runner)
        runner.logger.info(f'Frozen LiDAR integrity: all tensors unchanged at epoch {epoch}')
