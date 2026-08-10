"""Re-evaluate a persisted MMDetection3D KITTI prediction pickle."""

from __future__ import annotations

import argparse

from mmengine import load

from mmdet3d.evaluation.functional import kitti_eval
from mmdet3d.evaluation.metrics import KittiMetric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions')
    parser.add_argument(
        '--infos', default='data/kitti/kitti_infos_val.pkl')
    parser.add_argument(
        '--classes', nargs='+',
        default=['Pedestrian', 'Cyclist', 'Car'])
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = load(args.predictions)
    metric = KittiMetric(ann_file=args.infos)
    data_list = metric.convert_annos_to_kitti_annos(load(args.infos))
    if len(predictions) != len(data_list):
        raise RuntimeError(
            f'prediction count {len(predictions)} does not match '
            f'validation count {len(data_list)}')
    ground_truth = [item['kitti_annos'] for item in data_list]
    result_text, _ = kitti_eval(
        ground_truth,
        predictions,
        current_classes=args.classes,
        eval_types=['bbox', 'bev', '3d'])
    print(result_text)


if __name__ == '__main__':
    main()
