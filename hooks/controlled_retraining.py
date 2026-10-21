"""Provenance, exact warm-start, exposure and frozen-state controls."""

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import set_random_seed
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class ControlledRetrainingHook(Hook):
    """Fail closed on changed initialization, lost freezing or wrong budgets.

    This intentionally supports fresh model-only continuations, not resumes.
    Training is counted in actual optimizer steps, not nominal dataset epochs.
    """

    priority = 'LOW'
    lidar_names = ('voxel_encoder', 'middle_encoder', 'backbone', 'neck', 'bbox_head')
    added_prefixes = ('fusion_module.pyramid_queries.',
                      'fusion_module.pyramid_residuals.')
    reference_sha256 = '71e9086e8b5cc77ce3b4021b3cad639b085e2b52f213606385291eaa36b1e3b2'

    def __init__(self, reference_checkpoint, frozen_lidar=True,
                 expected_updates=1000, seed=0):
        self.reference_checkpoint = reference_checkpoint
        self.frozen_lidar = bool(frozen_lidar)
        self.expected_updates = int(expected_updates)
        self.seed = int(seed)
        self.steps = 0
        self.sample_order = hashlib.sha256()
        self.report = dict(status='initializing', first_samples=[], first_input_hashes=[])
        self.reference = None

    @staticmethod
    def _model(runner):
        return runner.model.module if is_model_wrapper(runner.model) else runner.model

    def _write(self, runner):
        path = Path(runner.work_dir) / 'controlled_report.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.report, indent=2) + '\n')
        temporary.replace(path)

    def before_train(self, runner):
        path = Path(self.reference_checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.reference_sha256:
            raise RuntimeError('Controlled retraining reference checkpoint hash changed')
        saved = torch.load(path, map_location='cpu')['state_dict']
        actual = self._model(runner).state_dict()
        added = set(actual) - set(saved)
        if set(saved) - set(actual) or any(
                not key.startswith(self.added_prefixes) for key in added):
            raise RuntimeError('Unexpected warm-start model keys')
        for key, value in saved.items():
            if not torch.equal(actual[key].detach().cpu(), value):
                raise RuntimeError(f'Common warm-start tensor changed: {key}')
        for key in added:
            value = actual[key].detach().cpu()
            if key.startswith('fusion_module.pyramid_queries.'):
                suffix = key.split('.', 3)[-1]
                if not torch.equal(value, saved['fusion_module.lidar_projection.' + suffix]):
                    raise RuntimeError(f'Additional query is not a cloned projection: {key}')
            elif torch.count_nonzero(value):
                raise RuntimeError(f'Additional output adapter is not zero: {key}')
        prefixes = tuple(name + '.' for name in self.lidar_names)
        self.reference = {k: v for k, v in saved.items() if k.startswith(prefixes)}
        if runner.optim_wrapper.optimizer.state:
            raise RuntimeError('Controlled arms require a fresh optimizer')
        self.step_handle = runner.optim_wrapper.optimizer.register_step_post_hook(
            self._after_optimizer_step)
        # Initialization/adapter construction must not change training RNG.
        set_random_seed(self.seed, deterministic=False)
        self.report.update(
            status='training', reference_checkpoint=str(path.resolve()),
            reference_sha256=digest, common_tensors_exact=True,
            added_tensors=sorted(added), fresh_optimizer=True,
            frozen_lidar=self.frozen_lidar, seed=self.seed,
            expected_optimizer_updates=self.expected_updates,
            accumulation=runner.optim_wrapper._accumulative_counts,
            lr_groups=[group['lr'] for group in runner.optim_wrapper.optimizer.param_groups],
            completed_microbatches=0, optimizer_updates=0)
        self._write(runner)

    def _after_optimizer_step(self, optimizer, args, kwargs):
        self.steps += 1

    def _check_frozen(self, runner, tensors=False):
        if not self.frozen_lidar:
            return
        model = self._model(runner)
        for name in self.lidar_names:
            module = getattr(model, name)
            if any(child.training for child in module.modules()):
                raise RuntimeError(f'Frozen LiDAR module is training: {name}')
            if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError(f'Frozen LiDAR module has gradients: {name}')
        if tensors:
            actual = model.state_dict()
            for key, value in self.reference.items():
                if not torch.equal(actual[key].detach().cpu(), value):
                    raise RuntimeError(f'Frozen LiDAR/head tensor changed: {key}')

    def before_train_epoch(self, runner):
        self._check_frozen(runner, tensors=True)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self._check_frozen(runner)
        ids = [str(sample.metainfo['sample_idx']) for sample in data_batch['data_samples']]
        self.sample_order.update((json.dumps(ids) + '\n').encode())
        if runner.iter < 16:
            self.report['first_samples'].append(ids)
        if runner.iter < 8:
            digest = hashlib.sha256()
            for key in ('imgs', 'img', 'points'):
                value = data_batch['inputs'].get(key)
                values = value if isinstance(value, (tuple, list)) else [value]
                for tensor in values:
                    if isinstance(tensor, torch.Tensor):
                        digest.update(key.encode())
                        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            self.report['first_input_hashes'].append(digest.hexdigest())

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not torch.isfinite(outputs['loss']).all():
            raise RuntimeError('Non-finite controlled training loss')
        if (runner.iter + 1) % 100 == 0:
            self.report.update(completed_microbatches=runner.iter + 1,
                               optimizer_updates=self.steps,
                               sample_order_sha256=self.sample_order.hexdigest(),
                               last_loss=float(outputs['loss'].detach()))
            if (runner.iter + 1) % 1000 == 0:
                self._check_frozen(runner, tensors=True)
            self._write(runner)

    def after_val_epoch(self, runner, metrics=None):
        self.report['validation_metrics'] = {k: float(v) for k, v in (metrics or {}).items()}
        self._write(runner)

    def after_train(self, runner):
        self._check_frozen(runner, tensors=True)
        if self.steps != self.expected_updates:
            raise RuntimeError(f'Expected {self.expected_updates} updates, got {self.steps}')
        self.report.update(status='complete', optimizer_updates=self.steps,
                           completed_microbatches=runner.iter,
                           sample_order_sha256=self.sample_order.hexdigest(),
                           frozen_tensors_unchanged=self.frozen_lidar)
        self.step_handle.remove()
        self._write(runner)


@HOOKS.register_module()
class ControlledModelSmokeHook(Hook):
    """Real-data initialization/prediction and added-scale backward checks.

    Used only in the eight-batch execution smokes, never in production arms.
    """

    priority = 'LOWEST'

    @staticmethod
    def _same(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert len(left) == len(right)
            for a, b in zip(left, right):
                ControlledModelSmokeHook._same(a, b)

    def before_train(self, runner):
        model = runner.model
        self.report = {}
        # A direct fixed dataset item does not consume the training sampler.
        # Restore augmentation/model RNGs before the actual controlled stream.
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                batch = runner.train_dataloader.collate_fn([
                    runner.train_dataloader.dataset[0]])
                self.processed = model.data_preprocessor(batch, training=True)
                model.eval()
                with torch.no_grad():
                    from projects.myfusion.fusion.mbt_nuscenes_multiscale import (
                        NuScenesMultiScaleMBTDetectorV2)
                    from projects.myfusion.fusion.mbt_detector_nuscenes_camera_v2 import (
                        NuScenesCameraFocusedMBTBEVFusionDetectorV2)
                    if isinstance(model, NuScenesMultiScaleMBTDetectorV2):
                        inputs = self.processed['inputs']
                        samples = self.processed['data_samples']
                        previous = NuScenesCameraFocusedMBTBEVFusionDetectorV2.extract_feat(
                            model, inputs, samples)
                        current = model.extract_feat(inputs, samples)
                        self._same(previous, current)
                        self._same(model.bbox_head(previous), model.bbox_head(current))
                        self.report['all_initial_features_and_head_logits_bitwise_equal'] = True
                    predictions = model(**self.processed, mode='predict')
                    result = predictions[0].pred_instances_3d
                    assert result.scores_3d.numel() > 0
                    assert torch.isfinite(result.scores_3d).all()
                    assert torch.isfinite(result.bboxes_3d.tensor).all()
                    self.report['finite_initial_detections'] = int(result.scores_3d.numel())
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            model.train()

    def after_train(self, runner):
        model = runner.model
        model.eval()
        module = model.fusion_module
        if hasattr(module, 'pyramid_residuals'):
            for adapter in module.pyramid_residuals.values():
                assert adapter.weight.detach().abs().sum() > 0
            features = model.extract_feat(
                self.processed['inputs'], self.processed['data_samples'])
            for level in (1, 2):
                scores = model.bbox_head.forward_single(features[level])[0]
                gradients = torch.autograd.grad(
                    scores.square().mean(),
                    (module.image_projection.weight, module.bottleneck_tokens),
                    retain_graph=level == 1)
                assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
            self.report['added_scale_adapters_updated'] = True
            self.report['both_added_scale_camera_and_bottleneck_gradients'] = True
        model.load_state_dict(torch.load(
            Path(runner.work_dir) / 'iter_8.pth', map_location='cpu')['state_dict'], strict=True)
        self.report.update(status='passed', strict_checkpoint_reload=True,
                           smoke_checkpoint_used_for_production=False)
        (Path(runner.work_dir) / 'model_smoke_report.json').write_text(
            json.dumps(self.report, indent=2) + '\n')
