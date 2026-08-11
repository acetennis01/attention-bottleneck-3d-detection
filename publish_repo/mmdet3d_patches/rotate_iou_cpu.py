"""Shapely reference implementation for KITTI rotated overlap tests."""

import numpy as np
from shapely.geometry import Polygon


def _validate_boxes(boxes, name):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(f'{name} must have shape (N, 5), got {boxes.shape}')
    return boxes


def _rect_poly(cx, cy, x_size, y_size, yaw):
    """Reproduce ``rbbox_to_corners`` from the official CUDA evaluator."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    hx = x_size / 2.0
    hy = y_size / 2.0
    points = np.array([
        [-hx, -hy],
        [-hx, hy],
        [hx, hy],
        [hx, -hy],
    ], dtype=np.float64)
    rotation = np.array([[c, s], [-s, c]], dtype=np.float64)
    points = points @ rotation.T
    points[:, 0] += cx
    points[:, 1] += cy
    return Polygon(points)


def rotate_iou_cpu_eval(boxes, query_boxes, criterion=-1):
    """Match MMDetection3D's ``rotate_iou_gpu_eval`` semantics on CPU."""
    boxes = _validate_boxes(boxes, 'boxes')
    query_boxes = _validate_boxes(query_boxes, 'query_boxes')
    overlaps = np.zeros(
        (boxes.shape[0], query_boxes.shape[0]), dtype=np.float64)
    first = [_rect_poly(*box) for box in boxes]
    second = [_rect_poly(*box) for box in query_boxes]
    for row, first_polygon in enumerate(first):
        area1 = first_polygon.area
        for column, second_polygon in enumerate(second):
            area2 = second_polygon.area
            intersection = first_polygon.intersection(second_polygon).area
            if intersection <= 0:
                continue
            if criterion == 0:
                overlaps[row, column] = intersection / area1
            elif criterion == 1:
                overlaps[row, column] = intersection / area2
            elif criterion == -1:
                overlaps[row, column] = (
                    intersection / (area1 + area2 - intersection))
            else:
                overlaps[row, column] = intersection
    return overlaps
