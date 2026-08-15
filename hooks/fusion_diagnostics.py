"""Low-overhead diagnostics for multimodal bottleneck training."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class FusionDiagnosticsHook(Hook):
    """Log component gradient norms and fusion-output magnitudes.

    Gradients are sampled only at the configured interval.  This makes it
    possible to detect a dead image branch, bottleneck collapse, or an
    oversized residual update without retaining activations between steps.
    """

    priority = 'NORMAL'

    def __init__(self, interval: int = 500) -> None:
        if interval < 1:
            raise ValueError('interval must be positive')
        self.interval = interval
        self._runner = None
        self._handles = []
        self._grad_squares: Dict[str, torch.Tensor] = {}
        self._feature_stats: Dict[str, float] = {}

    @staticmethod
    def _component(name: str) -> str:
        if name.startswith('img_backbone.'):
            return 'image_backbone'
        if name.startswith('fusion_module.image'):
            return 'fusion_image'
        if name.startswith('fusion_module.lidar'):
            return 'fusion_lidar'
        if name.startswith('fusion_module.bottleneck'):
            return 'bottleneck'
        if name.startswith(('fusion_module.readout',
                            'fusion_module.bev_residual')):
            return 'fusion_output'
        if name.startswith(('voxel_encoder.', 'middle_encoder.',
                            'backbone.', 'neck.')):
            return 'lidar_encoder'
        if name.startswith('bbox_head.'):
            return 'bbox_head'
        return 'other'

    def _is_logging_step(self) -> bool:
        return self._runner is not None and self.every_n_train_iters(
            self._runner, self.interval)

    def _grad_hook(self, component: str):
        def capture(gradient: torch.Tensor) -> None:
            if not self._is_logging_step():
                return
            squared = gradient.detach().float().square().sum()
            if component in self._grad_squares:
                self._grad_squares[component] += squared
            else:
                self._grad_squares[component] = squared
        return capture

    def _forward_hook(self, _module, inputs, output) -> None:
        if not self._is_logging_step():
            return
        lidar_features = inputs[1].detach().float()
        fused_features = output['bev_features'].detach().float()
        delta = fused_features - lidar_features

        def rms(tensor: torch.Tensor) -> float:
            return tensor.square().mean().sqrt().item()

        lidar_rms = rms(lidar_features)
        delta_rms = rms(delta)
        self._feature_stats = {
            'fusion/lidar_rms': lidar_rms,
            'fusion/bev_delta_rms': delta_rms,
            'fusion/bev_delta_ratio': delta_rms / max(lidar_rms, 1e-12),
            'fusion/image_token_rms': rms(output['image_tokens']),
            'fusion/lidar_token_rms': rms(output['lidar_tokens']),
            'fusion/bottleneck_token_rms': rms(
                output['bottleneck_tokens']),
        }

    def before_train(self, runner) -> None:
        self._runner = runner
        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        if not hasattr(model, 'fusion_module'):
            raise AttributeError(
                'FusionDiagnosticsHook requires model.fusion_module')

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                handle = parameter.register_hook(
                    self._grad_hook(self._component(name)))
                self._handles.append(handle)
        self._handles.append(
            model.fusion_module.register_forward_hook(self._forward_hook))

    def after_train_iter(self, runner, batch_idx: int,
                         data_batch=None, outputs=None) -> None:
        if not self.every_n_train_iters(runner, self.interval):
            return

        stats = dict(self._feature_stats)
        for component, squared in self._grad_squares.items():
            stats[f'grad_norm/{component}'] = squared.sqrt().item()

        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        stats['fusion/bev_residual_weight_norm'] = (
            model.fusion_module.bev_residual.weight.detach().float().norm().item())

        for name, value in sorted(stats.items()):
            runner.message_hub.update_scalar(f'train/{name}', value)
        formatted = ' '.join(
            f'{name}={value:.6g}' for name, value in sorted(stats.items()))
        runner.logger.info(f'Fusion diagnostics: {formatted}')
        self._grad_squares = {}
        self._feature_stats = {}

    def after_train(self, runner) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._runner = None

