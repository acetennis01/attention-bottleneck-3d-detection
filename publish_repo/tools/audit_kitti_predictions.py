"""Audit persisted KITTI predictions for common 3D box convention errors."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--predictions',
        default='work_dirs/mbt_bev_diagnostic/pred_instances_3d.pkl')
    parser.add_argument(
        '--infos', default='data/kitti/kitti_infos_val.pkl')
    parser.add_argument('--score-threshold', type=float, default=0.1)
    parser.add_argument('--horizontal-match-metres', type=float, default=2.0)
    return parser.parse_args()


def _load(path):
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def _ground_truth_by_sample(infos):
    categories = infos['metainfo']['categories']
    label_to_name = {value: key for key, value in categories.items()}
    result = defaultdict(lambda: defaultdict(list))
    for sample in infos['data_list']:
        sample_idx = int(sample['sample_idx'])
        for instance in sample.get('instances', []):
            label = int(instance['bbox_label'])
            name = label_to_name.get(label)
            if name in {'Car', 'Pedestrian', 'Cyclist'}:
                result[sample_idx][name].append(
                    np.asarray(instance['bbox_3d'], dtype=np.float64))
    return result


def _summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 'n=0'
    return (f'n={values.size} median={np.median(values):.3f} '
            f'p90={np.percentile(values, 90):.3f} mean={values.mean():.3f}')


def main():
    args = parse_args()
    predictions = _load(args.predictions)
    infos = _load(args.infos)
    ground_truth = _ground_truth_by_sample(infos)

    expected = len(infos['data_list'])
    if len(predictions) != expected:
        raise RuntimeError(
            f'prediction count {len(predictions)} != validation count {expected}')

    seen = set()
    stats = defaultdict(lambda: defaultdict(list))
    for sample in predictions:
        sample_ids = np.asarray(sample['sample_idx']).reshape(-1)
        if sample_ids.size == 0:
            continue
        sample_idx = int(sample_ids[0])
        if sample_idx in seen:
            raise RuntimeError(f'duplicate sample_idx {sample_idx}')
        seen.add(sample_idx)

        names = np.asarray(sample['name'])
        scores = np.asarray(sample['score'], dtype=np.float64)
        locations = np.asarray(sample['location'], dtype=np.float64)
        dimensions = np.asarray(sample['dimensions'], dtype=np.float64)
        rotations = np.asarray(sample['rotation_y'], dtype=np.float64)
        if not (np.isfinite(scores).all() and np.isfinite(locations).all()
                and np.isfinite(dimensions).all()
                and np.isfinite(rotations).all()):
            raise RuntimeError(f'non-finite prediction in sample {sample_idx}')
        if dimensions.size and np.any(dimensions <= 0):
            raise RuntimeError(f'non-positive dimensions in sample {sample_idx}')

        for class_name in ('Car', 'Pedestrian', 'Cyclist'):
            mask = ((names == class_name)
                    & (scores >= args.score_threshold))
            class_locations = locations[mask]
            class_dimensions = dimensions[mask]
            for dim_name, column in zip(('length', 'height', 'width'), range(3)):
                stats[class_name][dim_name].extend(class_dimensions[:, column])

            gt_boxes = ground_truth[sample_idx].get(class_name, [])
            if not gt_boxes or class_locations.size == 0:
                continue
            gt_boxes = np.asarray(gt_boxes)
            distances = np.linalg.norm(
                class_locations[:, None, [0, 2]]
                - gt_boxes[None, :, [0, 2]], axis=-1)
            nearest = distances.argmin(axis=1)
            best = distances[np.arange(len(class_locations)), nearest]
            close = best < args.horizontal_match_metres
            if not close.any():
                continue

            matched_gt = gt_boxes[nearest[close]]
            matched_locations = class_locations[close]
            matched_dimensions = class_dimensions[close]
            stats[class_name]['horizontal_error'].extend(best[close])
            stats[class_name]['bottom_y_error'].extend(
                np.abs(matched_locations[:, 1] - matched_gt[:, 1]))
            pred_top = matched_locations[:, 1] - matched_dimensions[:, 1]
            gt_top = matched_gt[:, 1] - matched_gt[:, 4]
            vertical_intersection = np.maximum(
                0.0,
                np.minimum(matched_locations[:, 1], matched_gt[:, 1])
                - np.maximum(pred_top, gt_top))
            stats[class_name]['vertical_intersection'].extend(
                vertical_intersection)

    print(f'validated {len(seen)}/{expected} prediction records')
    for class_name in ('Car', 'Pedestrian', 'Cyclist'):
        print(f'\n{class_name}')
        for key in ('length', 'height', 'width', 'horizontal_error',
                    'bottom_y_error', 'vertical_intersection'):
            print(f'  {key}: {_summarize(stats[class_name][key])}')


if __name__ == '__main__':
    main()
