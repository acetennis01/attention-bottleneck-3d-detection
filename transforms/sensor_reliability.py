"""KITTI sensor degradations with camera/LiDAR severity targets.

Targets are generated only from training augmentation. At inference they may
be recorded for score assessment, but the detector never reads them to gate.
"""

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS


CONDITIONS = ('clean', 'camera_dim', 'camera_blur', 'lidar_dropout', 'lidar_noise')


def corrupt_sensor_inputs(image, points, condition, severity, generator):
    if condition not in CONDITIONS or not 0 <= severity <= 1:
        raise ValueError('Unknown sensor condition or severity outside [0, 1]')
    camera_quality, lidar_quality = 1.0, 1.0
    image, points = image.copy(), points.clone()
    if severity == 0:
        return image, points, (camera_quality, lidar_quality)
    if condition.startswith('camera_'):
        camera_quality = 1.0 - severity
        if condition == 'camera_dim':
            image = (image.astype(np.float32) * (1.0 - severity)).astype(image.dtype)
        elif severity > 0:
            image = cv2.GaussianBlur(image, (0, 0), 12.0 * severity)
    elif condition.startswith('lidar_'):
        lidar_quality = 1.0 - severity
        if condition == 'lidar_dropout':
            # Retain one point to keep the existing pillar encoder contract.
            retained = max(1, round(len(points) * (1.0 - severity)))
            indices = generator.permutation(len(points))[:retained]
            points = points[indices]
        else:
            noise = generator.normal(0, 0.20 * severity, (len(points), 3))
            points.tensor[:, :3] += points.tensor.new_tensor(noise)
    return image, points, (camera_quality, lidar_quality)


@TRANSFORMS.register_module()
class SensorReliabilityCorruption(BaseTransform):
    """Apply after image/point loading and before Pack3DDetInputs.

    Clean samples occur with probability 0.25. Otherwise a single sensor and
    corruption are selected uniformly, with severity sampled from Beta(2, 2).
    Fixed evaluation corruptions use a sample-derived deterministic generator.
    """

    def __init__(self, condition='random', severity=0.0, seed=0):
        if condition not in (*CONDITIONS, 'random') or not 0 <= severity <= 1:
            raise ValueError('Unknown sensor condition or severity outside [0, 1]')
        self.condition, self.severity, self.seed = condition, float(severity), seed

    def transform(self, results):
        condition, severity = self.condition, self.severity
        if condition == 'random':
            generator = np.random
            if generator.random() < 0.25:
                condition, severity = 'clean', 0.0
            else:
                condition = generator.choice(CONDITIONS[1:])
                severity = float(generator.beta(2, 2))
        else:
            generator = np.random.default_rng(
                np.random.SeedSequence([self.seed, int(results['sample_idx'])]))
        image, points, target = corrupt_sensor_inputs(
            results['img'], results['points'], condition, severity, generator)
        results.update(img=image, points=points, modality_quality_target=target,
                       sensor_condition=condition, sensor_severity=severity)
        return results
