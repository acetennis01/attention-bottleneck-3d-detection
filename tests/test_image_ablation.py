"""Tests for deterministic camera-input ablations."""

from pathlib import Path

import numpy as np

from projects.myfusion.transforms.image_ablation import (
    BlackImage, ShuffleKittiImagePath, ShuffleNuScenesImagePaths)


def test_black_image_preserves_shape_and_dtype():
    image = np.full((12, 20, 3), 127, dtype=np.uint8)
    results = BlackImage().transform(dict(img=image))

    assert results['img'].shape == image.shape
    assert results['img'].dtype == image.dtype
    assert not results['img'].any()
    assert results['camera_ablation'] == 'black_image'


def test_black_image_supports_multiview_lists():
    images = [
        np.full((8, 10, 3), value, dtype=np.uint8)
        for value in (32, 96, 160)
    ]
    results = BlackImage().transform(dict(img=images))

    assert isinstance(results['img'], list)
    assert [image.shape for image in results['img']] == [
        image.shape for image in images]
    assert not any(image.any() for image in results['img'])


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


def test_nuscenes_shuffle_is_deterministic_and_preserves_camera_slots(
        tmp_path: Path):
    camera_names = ('CAM_FRONT', 'CAM_BACK')
    data_list = []
    for sample_idx in range(4):
        data_list.append(dict(
            sample_idx=sample_idx,
            images={
                camera: dict(img_path=f'{camera}_{sample_idx}.jpg')
                for camera in camera_names
            }))
    info_file = tmp_path / 'infos.pkl'
    import mmengine
    mmengine.dump(dict(data_list=data_list), info_file)

    first = ShuffleNuScenesImagePaths(str(info_file), seed=0)
    second = ShuffleNuScenesImagePaths(str(info_file), seed=0)
    assert first.mapping == second.mapping
    assert all(source != target for source, target in first.mapping.items())

    results = first.transform(dict(
        sample_idx=0,
        img_path=[
            '/data/samples/CAM_FRONT/CAM_FRONT_0.jpg',
            '/data/samples/CAM_BACK/CAM_BACK_0.jpg',
        ]))
    target = first.mapping[0]
    assert results['img_path'] == [
        f'/data/samples/CAM_FRONT/CAM_FRONT_{target}.jpg',
        f'/data/samples/CAM_BACK/CAM_BACK_{target}.jpg',
    ]

    nested = first.transform(dict(
        sample_idx=1,
        images={
            camera: dict(
                img_path=f'/data/samples/{camera}/{camera}_1.jpg',
                lidar2cam=f'calibration-{camera}')
            for camera in camera_names
        }))
    nested_target = first.mapping[1]
    for camera in camera_names:
        assert nested['images'][camera]['img_path'] == (
            f'/data/samples/{camera}/{camera}_{nested_target}.jpg')
        assert nested['images'][camera]['lidar2cam'] == (
            f'calibration-{camera}')
