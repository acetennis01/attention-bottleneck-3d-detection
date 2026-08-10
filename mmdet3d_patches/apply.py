"""Install the validated MMCV KITTI overlap backend into MMDetection3D."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


REPLACEMENTS = (
    (
        "    from .rotate_iou import rotate_iou_gpu_eval\n"
        "    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)\n",
        "    # The stock Numba kernel cannot compile when its CUDA runtime "
        "emits newer\n"
        "    # PTX than the driver accepts. MMCV's compiled operator is "
        "compatible with\n"
        "    # this environment and is regression-tested against the CPU "
        "reference.\n"
        "    from .rotate_iou_mmcv import rotate_iou_mmcv_eval\n"
        "    riou = rotate_iou_mmcv_eval(boxes, qboxes, criterion)\n",
    ),
    (
        "    from .rotate_iou import rotate_iou_gpu_eval\n"
        "    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],\n"
        "                               qboxes[:, [0, 2, 3, 5, 6]], 2)\n",
        "    from .rotate_iou_mmcv import rotate_iou_mmcv_eval\n"
        "    rinc = rotate_iou_mmcv_eval(boxes[:, [0, 2, 3, 5, 6]],\n"
        "                                qboxes[:, [0, 2, 3, 5, 6]], 2)\n",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mmdet3d-root', type=Path,
        default=Path(__file__).resolve().parents[3])
    return parser.parse_args()


def main():
    root = parse_args().mmdet3d_root.resolve()
    source_dir = Path(__file__).resolve().parent
    target_dir = (
        root / 'mmdet3d/evaluation/functional/kitti_utils')
    eval_path = target_dir / 'eval.py'
    if not eval_path.is_file():
        raise FileNotFoundError(f'MMDetection3D evaluator not found: {eval_path}')

    text = eval_path.read_text()
    for original, replacement in REPLACEMENTS:
        if replacement in text:
            continue
        if original not in text:
            raise RuntimeError(
                'Evaluator does not match the supported MMDetection3D 1.4.0 '
                'layout; refusing a partial modification.')
        text = text.replace(original, replacement, 1)

    shutil.copy2(source_dir / 'rotate_iou_mmcv.py', target_dir)
    shutil.copy2(source_dir / 'rotate_iou_cpu.py', target_dir)
    eval_path.write_text(text)
    print(f'Installed validated KITTI overlap backend in {target_dir}')


if __name__ == '__main__':
    main()
