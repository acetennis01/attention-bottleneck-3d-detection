"""Synthetic regression checks for matched camera-ablation summaries."""

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    '_kitti_summary_tests',
    Path(__file__).resolve().parents[1] / 'tools/summarize_kitti_camera_ablations.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_cases(root):
    for model, scores in [('lidar', {'correct': 38.}),
                          ('legacy_b8', {'correct': 43., 'black': 42., 'shuffled': 41.}),
                          ('v2_b8', {'correct': 44., 'black': 40., 'shuffled': 39.})]:
        for mode, score in scores.items():
            destination = root / model / mode
            destination.mkdir(parents=True)
            metrics = {
                'KITTI/Overall_3D_AP40_moderate': score,
                'KITTI/Overall_BEV_AP40_moderate': score + 10,
                **{'KITTI/' + label + '_3D_AP40_moderate_strict': score
                   for label in module.CLASSES}}
            (destination / 'metrics.json').write_text(json.dumps(dict(
                status='passed', validation_samples=3769,
                checkpoint_sha256=model, metrics=metrics)))


def test_deltas_are_correct_minus_corruption_and_baseline(tmp_path):
    write_cases(tmp_path)
    result = module.summarize(tmp_path)
    assert result['deltas']['v2_b8']['correct_minus_black']['3D_AP40_moderate'] == 4
    assert result['deltas']['v2_b8']['correct_minus_shuffled']['3D_AP40_moderate'] == 5
    assert result['deltas']['v2_b8']['correct_minus_lidar']['3D_AP40_moderate'] == 6
    assert result['deltas']['legacy_b8']['correct_minus_black']['3D_AP40_moderate'] == 1
    assert '| V2 (8 tokens) | correct | 44.0000 | 54.0000 |' in module.markdown(result)


@pytest.mark.parametrize('change', [dict(status='failed'),
                                   dict(validation_samples=32),
                                   dict(checkpoint_sha256='different')])
def test_rejects_incomplete_or_unmatched_cases(tmp_path, change):
    write_cases(tmp_path)
    source = tmp_path / 'v2_b8/black/metrics.json'
    report = json.loads(source.read_text())
    report.update(change)
    source.write_text(json.dumps(report))
    with pytest.raises(RuntimeError):
        module.summarize(tmp_path)
