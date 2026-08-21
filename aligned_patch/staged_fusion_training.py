"""Staged optimization that prevents immediate LiDAR shortcut learning."""

from __future__ import annotations

from typing import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class StagedFusionTrainingHook(Hook):
    """Freeze pretrained LiDAR and detection modules for initial epochs.

    Parameters stay in the optimizer and are re-enabled without rebuilding it.
    Frozen modules are also put in evaluation mode so their BatchNorm running
    statistics remain unchanged during the camera/fusion warm-up.
    """

    priority = 'HIGH'

    def __init__(
        self,
        freeze_epochs: int = 5,
        module_names: Sequence[str] = (
            'voxel_encoder',
            'middle_encoder',
            'backbone',
            'neck',
            'bbox_head',
        ),
    ) -> None:
        if freeze_epochs < 0:
            raise ValueError('freeze_epochs must be non-negative')
        self.freeze_epochs = int(freeze_epochs)
        self.module_names = tuple(module_names)
        self._frozen = None

    @staticmethod
    def _model(runner):
        return (runner.model.module if is_model_wrapper(runner.model)
                else runner.model)

    def _set_frozen(self, runner, frozen: bool) -> None:
        model = self._model(runner)
        for name in self.module_names:
            module = getattr(model, name, None)
            if module is None:
                continue
            module.requires_grad_(not frozen)
            module.eval() if frozen else module.train()
        if self._frozen != frozen:
            state = 'Frozen' if frozen else 'Unfrozen'
            runner.logger.info(
                f'{state} staged modules: {", ".join(self.module_names)}')
        self._frozen = frozen

    def before_train_epoch(self, runner) -> None:
        self._set_frozen(runner, runner.epoch < self.freeze_epochs)

    def before_train_iter(self, runner, batch_idx: int,
                          data_batch=None) -> None:
        # Some training loops restore model.train() between hooks. Reassert
        # eval mode without repeating parameter traversal on every iteration.
        if self._frozen:
            model = self._model(runner)
            for name in self.module_names:
                module = getattr(model, name, None)
                if module is not None:
                    module.eval()

