"""Build road-scene candidate sheets for class-specific attention figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mmengine
import numpy as np

from projects.myfusion.tools.render_attention_maps import (
    CLASS_NAMES, bbox_iou, find_candidates)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--info-file', default='data/kitti/kitti_infos_val.pkl')
    parser.add_argument(
        '--predictions-file',
        default=(
            'work_dirs/mbt_bev_camera_focused_best_eval/predictions/'
            'pred_instances_3d.pkl'))
    parser.add_argument(
        '--output-dir', default='work_dirs/attention_candidate_sheets')
    parser.add_argument('--count', type=int, default=24)
    return parser.parse_args()


def prediction_score(prediction, class_name, target_box):
    names = np.asarray(prediction['name'])
    boxes = np.asarray(prediction['bbox'])
    scores = np.asarray(prediction['score'])
    mask = names == class_name
    overlaps = np.asarray([bbox_iou(target_box, box) for box in boxes[mask]])
    return float(scores[mask][int(overlaps.argmax())])


def main():
    args = parse_args()
    data_list = mmengine.load(args.info_file)['data_list']
    predictions = mmengine.load(args.predictions_file)
    groups = find_candidates(data_list, predictions, excluded_samples=())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thumb_width, thumb_height = 420, 160
    columns = 4
    for label, class_name in enumerate(CLASS_NAMES):
        thumbnails = []
        for example in groups[label][:args.count]:
            info = data_list[example.dataset_index]
            image_path = Path('data/kitti/training/image_2') / (
                f'{example.sample_idx:06d}.png')
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            x1, y1, x2, y2 = (round(value) for value in example.bbox_2d)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
            score = prediction_score(
                predictions[example.dataset_index], class_name,
                example.bbox_2d)
            scale = min(
                thumb_width / image.shape[1],
                (thumb_height - 28) / image.shape[0])
            resized = cv2.resize(
                image,
                (round(image.shape[1] * scale),
                 round(image.shape[0] * scale)),
                interpolation=cv2.INTER_AREA)
            thumb = np.full(
                (thumb_height, thumb_width, 3), 245, dtype=np.uint8)
            x = (thumb_width - resized.shape[1]) // 2
            thumb[28:28 + resized.shape[0], x:x + resized.shape[1]] = resized
            cv2.putText(
                thumb,
                f'{example.sample_idx:06d}  score={score:.3f}',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (20, 20, 20), 1, cv2.LINE_AA)
            thumbnails.append(thumb)

        while len(thumbnails) % columns:
            thumbnails.append(np.full_like(thumbnails[0], 255))
        rows = [
            np.concatenate(thumbnails[index:index + columns], axis=1)
            for index in range(0, len(thumbnails), columns)
        ]
        sheet = np.concatenate(rows, axis=0)
        output = output_dir / f'{class_name.lower()}_candidates.png'
        if not cv2.imwrite(str(output), sheet):
            raise RuntimeError(f'Could not write {output}')
        print(f'Wrote {output}')


if __name__ == '__main__':
    main()
