"""Apply the fixed pilot corruption after KITTI evaluation image resizing."""

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

from .quality_corruptions import FIT_CONDITIONS, degrade_camera


@TRANSFORMS.register_module()
class KittiQualityCorruption(BaseTransform):
    def __init__(self, condition):
        self.condition = condition

    def transform(self, results):
        results['img'] = degrade_camera(results['img'], self.condition, 0.65)
        return results


@TRANSFORMS.register_module()
class KittiRandomQualityCorruption(BaseTransform):
    """Apply one quality degradation with varying severity at train time."""

    def __init__(self, probability=0.5):
        if not 0 <= probability <= 1:
            raise ValueError('probability must be in [0, 1]')
        self.probability = probability

    def transform(self, results):
        if np.random.random() < self.probability:
            condition = np.random.choice(FIT_CONDITIONS[1:])
            strength = np.random.random()
            results['img'] = degrade_camera(results['img'], condition, strength)
        return results
