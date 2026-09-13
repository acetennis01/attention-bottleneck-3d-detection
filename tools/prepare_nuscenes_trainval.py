"""Generate MMDetection3D metadata for nuScenes v1.0 trainval only."""

from argparse import ArgumentParser
from pathlib import Path

from tools.dataset_converters import nuscenes_converter
from tools.dataset_converters.create_gt_database import (
    create_groundtruth_database,
)
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--extra-tag', default='nuscenes')
    parser.add_argument('--max-sweeps', type=int, default=10)
    args = parser.parse_args()

    root_path = str(Path(args.root_path).resolve())
    nuscenes_converter.create_nuscenes_infos(
        root_path,
        args.extra_tag,
        version='v1.0-trainval',
        max_sweeps=args.max_sweeps,
    )

    train_info = f'{args.extra_tag}_infos_train.pkl'
    val_info = f'{args.extra_tag}_infos_val.pkl'
    update_pkl_infos(
        'nuscenes',
        out_dir=root_path,
        pkl_path=str(Path(root_path) / train_info),
    )
    update_pkl_infos(
        'nuscenes',
        out_dir=root_path,
        pkl_path=str(Path(root_path) / val_info),
    )
    create_groundtruth_database(
        'NuScenesDataset',
        root_path,
        args.extra_tag,
        train_info,
    )


if __name__ == '__main__':
    main()
