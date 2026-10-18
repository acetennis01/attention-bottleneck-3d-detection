"""Gradually enable coupled LiDAR masking after adapter initialization."""

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class CameraV2MaskingHook(Hook):
    """Epoch-indexed schedule reconstructed exactly when a run resumes.

    Epoch indices 0..warmup_epochs-1 use no masking. The next ramp_epochs
    increase linearly to max_drop_prob. This does not disable masking during
    later joint fine-tuning. The fusion module itself disables it in eval mode.
    """

    def __init__(self, max_drop_prob: float = 0.10,
                 warmup_epochs: int = 2, ramp_epochs: int = 3) -> None:
        if not 0 <= max_drop_prob < 1:
            raise ValueError('max_drop_prob must be in [0, 1)')
        if warmup_epochs < 0 or ramp_epochs < 1:
            raise ValueError('warmup must be non-negative and ramp positive')
        self.max_drop_prob = float(max_drop_prob)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)

    def before_train_epoch(self, runner) -> None:
        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        progress = min(1.0, max(
            0.0, (runner.epoch - self.warmup_epochs + 1) / self.ramp_epochs))
        module = model.fusion_module
        if not hasattr(module, 'lidar_spatial_drop_prob'):
            raise TypeError('CameraV2MaskingHook requires the V2 fusion module')
        module.lidar_spatial_drop_prob = self.max_drop_prob * progress
        runner.logger.info(
            'V2 coupled LiDAR spatial masking probability: '
            f'{module.lidar_spatial_drop_prob:.4f}')
