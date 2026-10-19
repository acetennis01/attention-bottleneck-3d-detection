"""Summarize a matched KITTI baseline and two camera-ablation triples."""

import argparse
import json
from pathlib import Path


MODELS = ('legacy_b8', 'v2_b8')
MODES = ('correct', 'black', 'shuffled')
CLASSES = ('Pedestrian', 'Cyclist', 'Car')


def metric(report, suffix):
    matches = [value for name, value in report['metrics'].items()
               if name.endswith('/' + suffix)]
    if len(matches) != 1:
        raise ValueError(f'Expected one metric ending in {suffix}: {matches}')
    return float(matches[0])


def summarize(root):
    reports = {}
    for model, modes in [('lidar', ('correct',)),
                         *[(name, MODES) for name in MODELS]]:
        reports[model] = {}
        for mode in modes:
            report = json.loads((root / model / mode / 'metrics.json').read_text())
            if report['status'] != 'passed' or report['validation_samples'] != 3769:
                raise RuntimeError(f'Incomplete full-validation case: {model}/{mode}')
            reports[model][mode] = report
        if len({report['checkpoint_sha256']
                for report in reports[model].values()}) != 1:
            raise RuntimeError(f'Corruptions used different checkpoints: {model}')
    result = dict(validation_samples=3769, overall={}, per_class={}, deltas={})
    for model, modes in reports.items():
        result['overall'][model] = {}
        result['per_class'][model] = {}
        for mode, report in modes.items():
            result['overall'][model][mode] = {
                '3D_AP40_moderate': metric(report, 'Overall_3D_AP40_moderate'),
                'BEV_AP40_moderate': metric(report, 'Overall_BEV_AP40_moderate')}
            result['per_class'][model][mode] = {
                label: metric(report, label + '_3D_AP40_moderate_strict')
                for label in CLASSES}
    for model in MODELS:
        rows = result['overall'][model]
        result['deltas'][model] = {}
        for comparison in ('black', 'shuffled', 'lidar'):
            other = (result['overall']['lidar']['correct'] if comparison == 'lidar'
                     else rows[comparison])
            result['deltas'][model]['correct_minus_' + comparison] = {
                key: rows['correct'][key] - other[key] for key in rows['correct']}
    return result


def markdown(summary):
    lines = [
        '# KITTI camera ablations: matched full-validation evaluation', '',
        'All results use 3,769 validation frames, fixed selected checkpoints, '
        'and the same MMCV KITTI overlap backend. Scores are AP points, not '
        'classification accuracy. No weights were updated during evaluation.', '',
        '| Model | Camera input | 3D AP40 moderate | BEV AP40 moderate |',
        '|---|---|---:|---:|']
    names = {'lidar': 'LiDAR-only', 'legacy_b8': 'Previous MBT (8 tokens)',
             'v2_b8': 'V2 (8 tokens)'}
    for model, modes in summary['overall'].items():
        for mode, row in modes.items():
            label = 'N/A' if model == 'lidar' else mode
            lines.append(f"| {names[model]} | {label} | "
                         f"{row['3D_AP40_moderate']:.4f} | "
                         f"{row['BEV_AP40_moderate']:.4f} |")
    lines += ['', '## Correct-image advantage', '',
              'Positive values mean correct images scored higher than the '
              'comparison. Differences are percentage points, not relative percent.', '',
              '| Model | Comparison | 3D difference | BEV difference |',
              '|---|---|---:|---:|']
    for model, comparisons in summary['deltas'].items():
        for comparison, row in comparisons.items():
            lines.append(f"| {names[model]} | {comparison.replace('_', ' ')} | "
                         f"{row['3D_AP40_moderate']:+.4f} | "
                         f"{row['BEV_AP40_moderate']:+.4f} |")
    lines += ['', '## Per-class strict moderate 3D AP40', '',
              '| Model | Camera | Pedestrian | Cyclist | Car |',
              '|---|---|---:|---:|---:|']
    for model, modes in summary['per_class'].items():
        for mode, row in modes.items():
            label = 'N/A' if model == 'lidar' else mode
            lines.append(f"| {names[model]} | {label} | " +
                         ' | '.join(f'{row[name]:.4f}' for name in CLASSES) + ' |')
    lines += ['', '## Interpretation limits', '',
              '- Black/shuffled images are corruption tests. A larger drop alone '
              'does not establish a better model; clean-image performance matters.',
              '- The fusion runs also fine-tune LiDAR weights. Their full gain '
              'over the transferred baseline is not isolated proof of camera benefit.',
              '- Auxiliary-task gradients and attention maps are not substitutes '
              'for these detection-metric ablations.',
              '- These are single-seed, validation-selected results. Statistical '
              'significance and generalization beyond this split are not established.',
              '- The active sixteen-token V2 training was not evaluated or paused '
              'by this analysis.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summary = summarize(args.root)
    output = args.root / 'report'
    output.mkdir(exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'SUMMARY.md').write_text(markdown(summary))
    print(markdown(summary))


if __name__ == '__main__':
    main()
