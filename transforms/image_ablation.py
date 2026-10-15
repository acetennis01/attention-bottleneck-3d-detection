"""Deterministic camera-input ablations for KITTI and nuScenes."""

from pathlib import Path
from typing import Dict, List

import mmengine
import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BlackImage(BaseTransform):
    """Replace a loaded image with an all-zero image of the same shape."""

    def transform(self, results: Dict) -> Dict:
        if 'img' not in results:
            raise KeyError('BlackImage must run after image loading')
        images = results['img']
        if isinstance(images, (list, tuple)):
            results['img'] = [np.zeros_like(image) for image in images]
        else:
            results['img'] = np.zeros_like(images)
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


@TRANSFORMS.register_module()
class ShuffleNuScenesImagePaths(BaseTransform):
    """Replace all six camera images with another validation sample.

    A seeded Sattolo cycle guarantees that no sample keeps its own images.
    LiDAR, camera calibration, and annotations remain tied to the original
    sample, which destroys image/scene correspondence without changing the
    geometric projection used by the detector.
    """

    def __init__(self, info_file: str, seed: int = 0) -> None:
        bundle = mmengine.load(info_file)
        data_list = bundle.get('data_list', bundle)
        if len(data_list) < 2:
            raise ValueError('nuScenes shuffling requires at least 2 samples')

        sample_ids = [int(info['sample_idx']) for info in data_list]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError('nuScenes info file contains duplicate sample IDs')
        shuffled_ids = sample_ids.copy()
        generator = np.random.default_rng(seed)
        for index in range(len(shuffled_ids) - 1, 0, -1):
            swap_index = int(generator.integers(0, index))
            shuffled_ids[index], shuffled_ids[swap_index] = (
                shuffled_ids[swap_index], shuffled_ids[index])
        if any(source == target for source, target in zip(
                sample_ids, shuffled_ids)):
            raise RuntimeError('Sattolo permutation unexpectedly has a fixed point')

        images_by_id = {
            int(info['sample_idx']): {
                camera: image['img_path']
                for camera, image in info['images'].items()
            }
            for info in data_list
        }
        self.mapping = dict(zip(sample_ids, shuffled_ids))
        self.images_by_id = images_by_id

    @staticmethod
    def _camera_name(path: Path, camera_names: List[str]) -> str:
        if path.parent.name in camera_names:
            return path.parent.name
        matches = [name for name in camera_names if name in path.name]
        if len(matches) != 1:
            raise ValueError(f'Cannot infer nuScenes camera from {path}')
        return matches[0]

    def transform(self, results: Dict) -> Dict:
        if 'sample_idx' not in results:
            raise KeyError('ShuffleNuScenesImagePaths requires sample_idx')
        source_id = int(results['sample_idx'])
        try:
            target_id = self.mapping[source_id]
            target_images = self.images_by_id[target_id]
        except KeyError as error:
            raise KeyError(
                f'Sample {source_id} is not present in the info file') from error

        if 'images' in results:
            original_paths = []
            shuffled_images = {}
            for camera, image_info in results['images'].items():
                image_info = image_info.copy()
                original_path = Path(image_info['img_path'])
                target_name = Path(target_images[camera]).name
                image_info['img_path'] = str(
                    original_path.with_name(target_name))
                original_paths.append(str(original_path))
                shuffled_images[camera] = image_info
            results['images'] = shuffled_images
        elif 'img_path' in results:
            paths = [Path(path) for path in results['img_path']]
            camera_names = list(target_images)
            original_paths = [str(path) for path in paths]
            results['img_path'] = [
                str(path.with_name(Path(target_images[
                    self._camera_name(path, camera_names)]).name))
                for path in paths
            ]
        else:
            raise KeyError(
                'ShuffleNuScenesImagePaths requires images or img_path')
        results['original_img_path'] = original_paths
        results['camera_ablation'] = 'shuffled_multiview_images'
        results['camera_ablation_target_sample_idx'] = target_id
        return results
