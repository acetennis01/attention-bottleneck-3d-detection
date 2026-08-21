"""Build a reproducible contact sheet from temporal-evaluation videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fraction', type=float, default=0.5)
    parser.add_argument('--width', type=int, default=1000)
    return parser.parse_args()


def read_frame(path: Path, fraction: float, width: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f'Could not open {path}')
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = round((frame_count - 1) * fraction)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f'Could not decode frame {frame_index} from {path}')
    height = round(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    if not 0 <= args.fraction <= 1:
        raise ValueError('--fraction must be between 0 and 1')
    frames = [read_frame(path, args.fraction, args.width)
              for path in args.videos]
    separator = np.full((12, args.width, 3), 255, dtype=np.uint8)
    parts = []
    for index, frame in enumerate(frames):
        if index:
            parts.append(separator)
        parts.append(frame)
    contact_sheet = np.concatenate(parts, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), contact_sheet):
        raise RuntimeError(f'Could not write {args.output}')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
