"""Fast rotated overlap using MMCV's compiled operator."""

import numpy as np
import torch
from mmcv.ops import box_iou_rotated


def rotate_iou_mmcv_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Match ``rotate_iou_gpu_eval`` using MMCV's supported CUDA operator."""
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(f'boxes must have shape (N, 5), got {boxes.shape}')
    if query_boxes.ndim != 2 or query_boxes.shape[1] != 5:
        raise ValueError(
            f'query_boxes must have shape (K, 5), got {query_boxes.shape}')
    if boxes.shape[0] == 0 or query_boxes.shape[0] == 0:
        return np.zeros(
            (boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)

    device = torch.device(
        f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    first = torch.from_numpy(boxes).to(device)
    second = torch.from_numpy(query_boxes).to(device)
    with torch.no_grad():
        # The legacy KITTI Numba kernel's numeric rotation matrix corresponds
        # to MMCV's counter-clockwise=False convention in Cartesian values.
        iou = box_iou_rotated(
            first, second, mode='iou', aligned=False,
            clockwise=False).clamp_(0.0, 1.0)
        if criterion == -1:
            result = iou
        else:
            area1 = (first[:, 2] * first[:, 3])[:, None]
            area2 = (second[:, 2] * second[:, 3])[None, :]
            intersection = iou * (area1 + area2) / (1.0 + iou)
            intersection = torch.minimum(
                intersection, torch.minimum(area1, area2))
            if criterion == 0:
                result = intersection / area1.clamp_min(1e-8)
            elif criterion == 1:
                result = intersection / area2.clamp_min(1e-8)
            else:
                result = intersection
    return result.cpu().numpy().astype(np.float32, copy=False)
