"""Deterministic camera-input ablations for KITTI evaluation."""

from pathlib import Path
from typing import Dict

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BlackImage(BaseTransform):
    """Replace a loaded image with an all-zero image of the same shape."""

    def transform(self, results: Dict) -> Dict:
        if 'img' not in results:
            raise KeyError('BlackImage must run after image loading')
        results['img'] = np.zeros_like(results['img'])
        results['camera_ablation'] = 'black_image'
        return results


@TRANSFORMS.register_module()
class ShuffleKittiImagePath(BaseTransform):
    """Replace each KITTI image with a different validation image.

    A seeded Sattolo cycle provides a deterministic permutation with no fixed
    points. LiDAR, calibration, and annotations remain tied to the original
    sample, so only the semantic correspondence of the camera image is broken.
    """

    def __init__(self, split_file: str, seed: int = 0) -> None:
        sample_ids = [
            line.strip() for line in Path(split_file).read_text().splitlines()
            if line.strip()
        ]
        if len(sample_ids) < 2:
            raise ValueError('The shuffled-image split needs at least 2 samples')
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError('The shuffled-image split contains duplicate IDs')

        shuffled_ids = sample_ids.copy()
        generator = np.random.default_rng(seed)
        for index in range(len(shuffled_ids) - 1, 0, -1):
            swap_index = int(generator.integers(0, index))
            shuffled_ids[index], shuffled_ids[swap_index] = (
                shuffled_ids[swap_index], shuffled_ids[index])
        self.mapping = dict(zip(sample_ids, shuffled_ids))
        if any(source == target for source, target in self.mapping.items()):
            raise RuntimeError('Sattolo permutation unexpectedly has a fixed point')

    def transform(self, results: Dict) -> Dict:
        if 'img_path' not in results:
            raise KeyError('ShuffleKittiImagePath requires img_path')
        original_path = Path(results['img_path'])
        try:
            shuffled_id = self.mapping[original_path.stem]
        except KeyError as error:
            raise KeyError(
                f'{original_path.stem} is not present in the configured split') \
                from error
        results['img_path'] = str(
            original_path.with_name(shuffled_id + original_path.suffix))
        results['camera_ablation'] = 'shuffled_image'
        results['original_img_path'] = str(original_path)
        return results
