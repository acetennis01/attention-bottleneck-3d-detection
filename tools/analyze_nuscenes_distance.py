"""Compute custom ego-distance-stratified nuScenes detection mAP.

The official nuScenes ``dist_ths`` values are center-distance matching
tolerances, not object-range bins.  This script first applies the standard
nuScenes class-range and point-count filters, then partitions predictions and
ground truth by radial ego distance.  Within each range bin it reuses the
official center-distance AP implementation and averages over matching
thresholds and classes that have eligible ground truth in that bin.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt,
    load_prediction,
)
from nuscenes.eval.detection.algo import accumulate, calc_ap
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox


Range = Tuple[float, Optional[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--eval-set', default='val')
    parser.add_argument('--baseline-results', required=True)
    parser.add_argument('--fusion-results', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def radial_distance(box: DetectionBox) -> float:
    return math.hypot(float(box.ego_translation[0]),
                      float(box.ego_translation[1]))


def in_range(distance: float, bounds: Range) -> bool:
    lower, upper = bounds
    return distance >= lower and (upper is None or distance < upper)


def subset_by_range(boxes: EvalBoxes, bounds: Range) -> EvalBoxes:
    subset = EvalBoxes()
    for sample_token in boxes.sample_tokens:
        subset.add_boxes(
            sample_token,
            [box for box in boxes[sample_token]
             if in_range(radial_distance(box), bounds)],
        )
    return subset


def class_gt_counts(boxes: EvalBoxes, class_names: Iterable[str]) -> Dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for box in boxes.all:
        counts[box.detection_name] += 1
    return counts


def evaluate_bin(gt_boxes: EvalBoxes, pred_boxes: EvalBoxes, cfg) -> dict:
    counts = class_gt_counts(gt_boxes, cfg.class_names)
    class_ap = {}
    threshold_ap = {}
    for class_name in cfg.class_names:
        if counts[class_name] == 0:
            continue
        aps = []
        threshold_ap[class_name] = {}
        for dist_th in cfg.dist_ths:
            metric_data = accumulate(
                gt_boxes,
                pred_boxes,
                class_name,
                cfg.dist_fcn_callable,
                dist_th,
            )
            ap = float(calc_ap(
                metric_data, cfg.min_recall, cfg.min_precision))
            threshold_ap[class_name][str(dist_th)] = ap
            aps.append(ap)
        class_ap[class_name] = float(np.mean(aps))

    return {
        'mAP': float(np.mean(list(class_ap.values()))),
        'num_evaluated_classes': len(class_ap),
        'num_ground_truth_boxes': int(sum(counts.values())),
        'ground_truth_by_class': counts,
        'class_ap': class_ap,
        'threshold_ap': threshold_ap,
    }


def prepare_ground_truth(nusc: NuScenes, cfg, eval_set: str) -> EvalBoxes:
    boxes = load_gt(nusc, eval_set, DetectionBox, verbose=False)
    boxes = add_center_dist(nusc, boxes)
    return filter_eval_boxes(
        nusc, boxes, cfg.class_range, verbose=False)


def prepare_predictions(nusc: NuScenes, cfg, result_path: str) -> EvalBoxes:
    boxes, _ = load_prediction(
        result_path,
        cfg.max_boxes_per_sample,
        DetectionBox,
        verbose=False,
    )
    boxes = add_center_dist(nusc, boxes)
    return filter_eval_boxes(
        nusc, boxes, cfg.class_range, verbose=False)


def main() -> None:
    args = parse_args()
    cfg = config_factory('detection_cvpr_2019')
    nusc = NuScenes(
        version=args.version, dataroot=args.data_root, verbose=False)
    gt_boxes = prepare_ground_truth(nusc, cfg, args.eval_set)

    methods = {
        'pointpillars': args.baseline_results,
        'abfuse': args.fusion_results,
    }
    predictions = {
        method: prepare_predictions(nusc, cfg, result_path)
        for method, result_path in methods.items()
    }
    for method, boxes in predictions.items():
        if set(boxes.sample_tokens) != set(gt_boxes.sample_tokens):
            raise ValueError(
                f'{method} predictions do not cover the evaluation split')

    bins: Dict[str, Range] = {
        'all': (0.0, None),
        'lt_15m': (0.0, 15.0),
        '15_to_30m': (15.0, 30.0),
        'gte_30m': (30.0, None),
    }
    output = {
        'protocol': {
            'description': (
                'Official nuScenes AP recomputed after radial ego-distance '
                'binning; classes without eligible GT in a bin are omitted.'),
            'matching_thresholds_m': list(cfg.dist_ths),
            'class_ranges_m': cfg.class_range,
            'bins_m': {
                name: [lower, upper] for name, (lower, upper) in bins.items()
            },
        },
        'methods': {},
    }

    for method, pred_boxes in predictions.items():
        output['methods'][method] = {}
        for name, bounds in bins.items():
            gt_subset = subset_by_range(gt_boxes, bounds)
            pred_subset = subset_by_range(pred_boxes, bounds)
            output['methods'][method][name] = evaluate_bin(
                gt_subset, pred_subset, cfg)

    for name in bins:
        baseline = output['methods']['pointpillars'][name]['mAP']
        fusion = output['methods']['abfuse'][name]['mAP']
        output['methods']['abfuse'][name]['delta_vs_pointpillars'] = (
            fusion - baseline)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + '\n')

    print('Distance bin          PointPillars   ABFuse   Delta   classes')
    for name in bins:
        baseline = output['methods']['pointpillars'][name]
        fusion = output['methods']['abfuse'][name]
        print(
            f'{name:20s} {baseline["mAP"]:.6f}     '
            f'{fusion["mAP"]:.6f}  '
            f'{fusion["delta_vs_pointpillars"]:+.6f}  '
            f'{fusion["num_evaluated_classes"]}')
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
