"""Finite-loss checks, actual optimizer-step counts and component gradients."""
import json
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet3d.registry import HOOKS

from ..fusion.nuimages_swin import (
    camera_state_for_backbone, NUIMAGES_SWIN_SHA256)
from ..tools.transfusion_l_checkpoint import sha256_file


@HOOKS.register_module()
class TransFusionLTrainingAuditHook(Hook):
    """Audit cold camera loading and native resumes; never modify weights.

    Component gradient norms are sampled in the optimizer's pre-step hook,
    AFTER the optimizer wrapper clips the accumulated gradients. They are not
    raw per-sample gradients and should not be compared to un-clipped norms.
    """
    priority = 'LOW'

    def __init__(self, interval=200,
                 camera_report='work_dirs/transfusion_l_mbt_swint_initialization/camera_initialization_verification.json'):
        self.interval = int(interval)
        if self.interval < 1:
            raise ValueError('interval must be positive')
        self.camera_report = camera_report

    def before_train(self, runner):
        self.runner = runner
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        source = json.loads(Path(self.camera_report).read_text())
        if (source['status'] != 'passed' or source['validation_frames'] != 6019
                or source['kind'] != 'zero_residual_initialization_only'):
            raise RuntimeError('Complete full pretrained-camera initialization validation first')
        path = runner.cfg.camera_checkpoint
        if (sha256_file(path) != NUIMAGES_SWIN_SHA256 or
                source['pretrained_camera']['sha256'] != NUIMAGES_SWIN_SHA256):
            raise RuntimeError('Camera initializer differs from the validated checkpoint')
        self.start = int(runner.iter)
        self.accumulation = runner.optim_wrapper._accumulative_counts
        if self.start % self.accumulation:
            raise RuntimeError('Resume splits a gradient-accumulation window')
        resumed = bool(runner.cfg.get('resume', False))
        if not resumed:
            if self.start or runner.optim_wrapper.optimizer.state:
                raise RuntimeError('A cold run requires iteration zero and a fresh optimizer')
            raw = torch.load(path, map_location='cpu')['state_dict']
            reference, _ = camera_state_for_backbone(model.img_backbone, raw)
            for key, value in model.img_backbone.state_dict().items():
                if not torch.equal(value.cpu(), reference[key]):
                    raise RuntimeError(f'Camera cold initialization changed: {key}')
        self.steps = 0
        self.gradients = {}
        self.groups = dict(camera=list(model.img_backbone.parameters()),
                           image_fpn=list(model.img_neck.parameters()),
                           fusion=list(model.fusion_module.parameters()),
                           bottleneck=[model.fusion_module.bottleneck_tokens])
        optimizer = runner.optim_wrapper.optimizer
        self.handles = [optimizer.register_step_pre_hook(self._before_step),
                        optimizer.register_step_post_hook(self._after_step)]
        self.report = dict(status='training', start_microbatch=self.start,
            resumed=resumed, accumulation=self.accumulation,
            optimizer_updates_this_session=0,
            total_optimizer_updates=self.start // self.accumulation,
            camera_checkpoint_sha256=NUIMAGES_SWIN_SHA256,
            exact_cold_camera_initialization=not resumed,
            initial_optimizer_state_entries=len(optimizer.state),
            camera_initialization_validation='passed',
            gradient_norm_definition='Accumulated gradients after global clipping, before optimizer.step')
        self._write()

    def _before_step(self, optimizer, args, kwargs):
        if self.steps >= 2 and (self.runner.iter + 1) % self.interval:
            return
        gradients = {}
        for name, parameters in self.groups.items():
            sums = [parameter.grad.detach().float().square().sum()
                    for parameter in parameters if parameter.grad is not None]
            norm = torch.stack(sums).sum().sqrt() if sums else torch.tensor(0.)
            if not torch.isfinite(norm):
                raise RuntimeError(f'Non-finite {name} gradient norm')
            gradients[name] = float(norm)
        self.gradients = gradients
        for name, value in gradients.items():
            self.runner.message_hub.update_scalar('train/grad_norm_postclip/' + name, value)

    def _after_step(self, optimizer, args, kwargs):
        self.steps += 1

    def _write(self):
        destination = Path(self.runner.work_dir) / 'training_progress.json'
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.report, indent=2) + '\n')
        temporary.replace(destination)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not torch.isfinite(torch.as_tensor(outputs['loss'])).all():
            raise RuntimeError('Non-finite training loss')
        if runner.iter < self.start + 8 or (runner.iter + 1) % self.interval == 0:
            self.report.update(completed_microbatches=runner.iter + 1,
                optimizer_updates_this_session=self.steps,
                total_optimizer_updates=self.start // self.accumulation + self.steps,
                gradient_norms_postclip=self.gradients,
                current_learning_rates=sorted(set(group['lr'] for group in
                    runner.optim_wrapper.optimizer.param_groups)))
            self._write()

    def after_train(self, runner):
        expected = (runner.iter - self.start) // self.accumulation
        if self.steps != expected:
            raise RuntimeError(f'Expected {expected} optimizer updates, got {self.steps}')
        self.report.update(status='complete', completed_microbatches=runner.iter,
            optimizer_updates_this_session=self.steps,
            total_optimizer_updates=self.start // self.accumulation + self.steps)
        self._write()
        for handle in self.handles:
            handle.remove()
