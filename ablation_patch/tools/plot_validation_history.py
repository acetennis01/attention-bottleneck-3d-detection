"""Extract and plot overall KITTI 3D AP40 moderate from training logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


PATTERN = re.compile(
    r'Epoch\(val\) \[(\d+)\].*?Overall_3D_AP40_moderate: ([0-9.]+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lidar-log', required=True, type=Path)
    parser.add_argument('--mbt-log', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    return parser.parse_args()


def read_history(path: Path) -> List[Tuple[int, float]]:
    history = []
    for line in path.read_text().splitlines():
        match = PATTERN.search(line)
        if match:
            history.append((int(match.group(1)), float(match.group(2))))
    if not history:
        raise ValueError(f'No validation AP40 entries found in {path}')
    return history


def summarize(history: List[Tuple[int, float]]) -> dict:
    best_epoch, best_ap = max(history, key=lambda item: item[1])
    return dict(
        validation_points=len(history),
        first_epoch=history[0][0],
        first_ap40_moderate=history[0][1],
        best_epoch=best_epoch,
        best_ap40_moderate=best_ap,
        final_epoch=history[-1][0],
        final_ap40_moderate=history[-1][1],
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    histories = {
        'LiDAR-only PointPillars': read_history(args.lidar_log),
        'Camera-LiDAR MBT': read_history(args.mbt_log),
    }

    epochs = sorted({epoch for history in histories.values()
                     for epoch, _ in history})
    by_label = {
        label: dict(history) for label, history in histories.items()
    }
    with (args.out_dir / 'validation_ap40_history.csv').open(
            'w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['epoch', *histories.keys()])
        for epoch in epochs:
            writer.writerow([
                epoch, *[by_label[label].get(epoch, '') for label in histories]
            ])

    summaries = {
        label: summarize(history) for label, history in histories.items()
    }
    (args.out_dir / 'validation_ap40_summary.json').write_text(
        json.dumps(summaries, indent=2) + '\n')

    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    colors = ('#355CDE', '#E07A20')
    for (label, history), color in zip(histories.items(), colors):
        x, y = zip(*history)
        axis.plot(x, y, marker='o', markersize=2.8, linewidth=1.8,
                  label=label, color=color)
        best_epoch, best_ap = max(history, key=lambda item: item[1])
        axis.scatter([best_epoch], [best_ap], s=55, color=color,
                     edgecolor='white', linewidth=1.0, zorder=3)
        axis.annotate(
            f'{best_ap:.2f} (epoch {best_epoch})',
            xy=(best_epoch, best_ap), xytext=(5, 7),
            textcoords='offset points', fontsize=8, color=color)

    axis.set_xlabel('Training epoch')
    axis.set_ylabel(r'Overall 3D AP$_{40}$ moderate')
    axis.set_title('KITTI validation convergence')
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(args.out_dir / 'validation_ap40_curve.png', dpi=200)
    fig.savefig(args.out_dir / 'validation_ap40_curve.pdf')

    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
