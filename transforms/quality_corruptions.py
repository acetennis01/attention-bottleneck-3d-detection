"""Deterministic image degradations for the KITTI quality-head pilot."""

import cv2
import numpy as np


FIT_CONDITIONS = ('correct', 'black', 'dim', 'blurred', 'occluded')


def degrade_camera(image, condition, strength):
    """Change pixel content while preserving image shape and calibration."""
    if condition == 'correct':
        return image
    if condition == 'black':
        return np.zeros_like(image)
    if condition == 'dim':
        return (image * (0.10 + 0.25 * (1 - strength))).astype(image.dtype)
    if condition == 'blurred':
        return cv2.GaussianBlur(image, (0, 0), 4 + 8 * strength)
    if condition == 'occluded':
        height, width = image.shape[:2]
        left = round(width * (0.15 + 0.12 * (1 - strength)))
        right = left + round(width * (0.30 + 0.15 * strength))
        result = image.copy()
        result[height // 8:7 * height // 8, left:right] = 0
        return result
    raise ValueError(f'Unknown quality condition: {condition}')
