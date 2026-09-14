"""Generate MMDetection3D metadata for nuScenes v1.0 trainval only."""

import os
from argparse import ArgumentParser
from pathlib import Path

from tools.dataset_converters import nuscenes_converter
from tools.dataset_converters.create_gt_database import (
    create_groundtruth_database,
)
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos
from mmdet3d.utils import register_all_modules


def main() -> None:
    register_all_modules(init_default_scope=True)

    parser = ArgumentParser()
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--extra-tag', default='nuscenes')
    parser.add_argument('--max-sweeps', type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root_path).resolve()
    root_path = str(root)
    train_info = f'{args.extra_tag}_infos_train.pkl'
    val_info = f'{args.extra_tag}_infos_val.pkl'
    train_path = root / train_info
    val_path = root / val_info
    state_dir = root / '.setup_state'
    state_dir.mkdir(exist_ok=True)

    if not train_path.is_file() or not val_path.is_file():
        nuscenes_converter.create_nuscenes_infos(
            root_path,
            args.extra_tag,
            version='v1.0-trainval',
            max_sweeps=args.max_sweeps,
        )
    else:
        print('Reusing existing train and validation info files.')

    # MMDetection3D 1.4.0 hardcodes ``./data/nuscenes`` while upgrading
    # nuScenes info files. Supply that path inside an isolated working
    # directory without changing the real project data directory.
    converter_cwd = root / '.mmdet3d_converter_cwd'
    compatibility_link = converter_cwd / 'data' / 'nuscenes'
    compatibility_link.parent.mkdir(parents=True, exist_ok=True)
    if compatibility_link.is_symlink():
        if compatibility_link.resolve() != root:
            raise RuntimeError(
                f'{compatibility_link} points to the wrong dataset')
    elif compatibility_link.exists():
        raise RuntimeError(f'{compatibility_link} exists and is not a symlink')
    else:
        compatibility_link.symlink_to(root, target_is_directory=True)

    original_cwd = Path.cwd()
    try:
        os.chdir(converter_cwd)
        for split, info_path in (('train', train_path), ('val', val_path)):
            marker = state_dir / f'{split}_infos_v2.complete'
            if marker.is_file():
                print(f'Skipping completed {split} info upgrade.')
                continue

            update_pkl_infos(
                'nuscenes', out_dir=root_path, pkl_path=str(info_path))
            marker.touch()
    finally:
        os.chdir(original_cwd)

    gt_marker = state_dir / 'groundtruth_database.complete'
    if not gt_marker.is_file():
        create_groundtruth_database(
            'NuScenesDataset',
            root_path,
            args.extra_tag,
            train_info,
        )
        gt_marker.touch()
    else:
        print('Skipping completed ground-truth database generation.')


if __name__ == '__main__':
    main()
