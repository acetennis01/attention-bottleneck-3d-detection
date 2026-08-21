"""Compare persisted KITTI predictions from camera-input ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import mmengine
import numpy as np


NUMERIC_FIELDS = (
    'alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidates', nargs='+', type=Path)
    return parser.parse_args()


def compare(reference: List[Dict], candidate: List[Dict]) -> Dict:
    if len(reference) != len(candidate):
        raise ValueError('Prediction files contain different frame counts')

    shape_mismatch_frames = 0
    name_mismatch_frames = 0
    changed_frames = 0
    total_detections = 0
    differences = {field: [] for field in NUMERIC_FIELDS}

    for expected, observed in zip(reference, candidate):
        expected_count = len(expected['name'])
        total_detections += expected_count
        if expected_count != len(observed['name']):
            shape_mismatch_frames += 1
            changed_frames += 1
            continue
        if not np.array_equal(expected['name'], observed['name']):
            name_mismatch_frames += 1
            changed_frames += 1
            continue

        frame_changed = False
        for field in NUMERIC_FIELDS:
            delta = np.abs(
                np.asarray(expected[field], dtype=np.float64) -
                np.asarray(observed[field], dtype=np.float64))
            differences[field].append(delta.reshape(-1))
            frame_changed |= bool(np.any(delta > 0))
        changed_frames += int(frame_changed)

    summary = dict(
        frames=len(reference),
        total_reference_detections=total_detections,
        shape_mismatch_frames=shape_mismatch_frames,
        name_mismatch_frames=name_mismatch_frames,
        numerically_changed_frames=changed_frames,
        fields={},
    )
    for field, chunks in differences.items():
        values = np.concatenate(chunks) if chunks else np.empty(0)
        summary['fields'][field] = dict(
            compared_values=int(values.size),
            mean_abs_difference=float(values.mean()) if values.size else None,
            max_abs_difference=float(values.max()) if values.size else None,
            values_over_1e_6=int(np.count_nonzero(values > 1e-6)),
            values_over_1e_4=int(np.count_nonzero(values > 1e-4)),
        )
    return summary


def main() -> None:
    args = parse_args()
    reference = mmengine.load(args.reference)
    for candidate_path in args.candidates:
        candidate = mmengine.load(candidate_path)
        print(f'\n{candidate_path}')
        print(json.dumps(compare(reference, candidate), indent=2))


if __name__ == '__main__':
    main()
