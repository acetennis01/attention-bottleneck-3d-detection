"""Do not train fusion until the public LiDAR baseline has been reproduced."""
import json
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet3d.registry import HOOKS

from ..tools.transfusion_l_checkpoint import (
    NEW_PREFIXES, SHA256, sha256_file, state_for_model, load_lidar_checkpoint)


@HOOKS.register_module()
class TransFusionLInitializationHook(Hook):
    priority = 'HIGH'

    def __init__(self, reference_checkpoint, baseline_report, interval=1000):
        self.reference_checkpoint = reference_checkpoint
        self.baseline_report = baseline_report
        self.interval = interval
        self.reference = None

    def before_train(self, runner):
        report = json.loads(Path(self.baseline_report).read_text())
        if report['status'] != 'passed' or report['validation_frames'] != 6019:
            raise RuntimeError('Reproduce the full LiDAR baseline before MBT training')
        if report['checkpoint']['sha256'] != SHA256 or sha256_file(self.reference_checkpoint) != SHA256:
            raise RuntimeError('Baseline and training must use the verified official checkpoint')
        annotation = runner.cfg.val_evaluator.ann_file
        if sha256_file(annotation) != report['annotation_sha256']:
            raise RuntimeError('Validation annotations changed since baseline reproduction')
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        # Explicitly load all LiDAR/head tensors on a cold start, but NEVER
        # overwrite a resumed detector before checking it for unwanted drift.
        if runner.iter == 0 and not runner.cfg.get('resume', False):
            transfer = load_lidar_checkpoint(model, self.reference_checkpoint, fusion=True)
            runner.logger.info('Audited TransFusion-L transfer: %s', transfer)
        raw = torch.load(self.reference_checkpoint, map_location='cpu')['state_dict']
        self.reference, _ = state_for_model(model, raw)
        self._verify(runner, initial=runner.iter == 0)

    def _verify(self, runner, initial=False):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        if not model.freeze_lidar:
            raise RuntimeError('This controlled experiment requires permanently frozen LiDAR/head')
        state = model.state_dict()
        protected = {key: value for key, value in state.items()
                     if not key.startswith(NEW_PREFIXES)}
        if set(self.reference) - set(protected):
            raise RuntimeError('Some reference LiDAR/head tensors are absent')
        for key, tensor in protected.items():
            reference = self.reference.get(key)
            if reference is None:
                if not key.endswith('num_batches_tracked') or torch.count_nonzero(tensor):
                    raise RuntimeError(f'Unexpected LiDAR state: {key}')
            elif not torch.equal(tensor.detach().cpu(), reference):
                raise RuntimeError(f'Pretrained LiDAR/head state changed: {key}')
        for name in model.lidar_module_names:
            module = getattr(model, name)
            if module.training or any(parameter.requires_grad for parameter in module.parameters()):
                raise RuntimeError(f'Pretrained module not fully frozen: {name}')
        if initial:
            for tensor in model.fusion_module.bev_residual.parameters():
                if torch.count_nonzero(tensor):
                    raise RuntimeError('Fresh fusion must start with zero residual output')
        record = dict(status='verified', iter=runner.iter,
            checkpoint_sha256=SHA256, protected_tensors=len(protected),
            lidar_and_head_unchanged=True, baseline_report=self.baseline_report)
        Path(runner.work_dir, 'transfusion_l_integrity.json').write_text(
            json.dumps(record, indent=2) + '\n')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if (runner.iter + 1) % self.interval == 0:
            self._verify(runner)

    def after_train(self, runner):
        self._verify(runner)
