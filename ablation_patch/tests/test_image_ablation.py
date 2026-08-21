"""Tests for deterministic camera-input ablations."""

from pathlib import Path

import numpy as np

from projects.myfusion.transforms.image_ablation import (
    BlackImage, ShuffleKittiImagePath)


def test_black_image_preserves_shape_and_dtype():
    image = np.full((12, 20, 3), 127, dtype=np.uint8)
    results = BlackImage().transform(dict(img=image))

    assert results['img'].shape == image.shape
    assert results['img'].dtype == image.dtype
    assert not results['img'].any()
    assert results['camera_ablation'] == 'black_image'


def test_shuffle_is_deterministic_and_has_no_fixed_points(tmp_path: Path):
    split = tmp_path / 'val.txt'
    split.write_text('000001\n000002\n000003\n000004\n')

    first = ShuffleKittiImagePath(str(split), seed=0)
    second = ShuffleKittiImagePath(str(split), seed=0)

    assert first.mapping == second.mapping
    assert all(source != target for source, target in first.mapping.items())
    results = first.transform(dict(img_path='/dataset/image_2/000001.png'))
    assert Path(results['img_path']).stem == first.mapping['000001']
    assert results['original_img_path'].endswith('000001.png')
