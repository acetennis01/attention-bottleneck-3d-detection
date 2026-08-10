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
    pts = np.array([
        [-hx, -hy],
        [-hx, hy],
        [hx, hy],
        [hx, -hy],
    ], dtype=np.float64)
    # Positive KITTI camera yaw is clockwise in the x-z plane.
    R = np.array([[c, s], [-s, c]], dtype=np.float64)
    pts = pts @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return Polygon(pts)

def rotate_iou_cpu_eval(boxes, query_boxes, criterion=-1):
    """Match MMDetection3D's ``rotate_iou_gpu_eval`` semantics on CPU.

    Criteria ``-1``, ``0`` and ``1`` return IoU/intersection fractions.  Any
    other value, notably ``2`` used by KITTI 3D evaluation, must return raw
    intersection area.  Treating criterion 2 as IoU corrupts the subsequent
    volume calculation and can collapse 3D AP to zero while BEV AP looks good.
    """
    boxes = _validate_boxes(boxes, 'boxes')
    query_boxes = _validate_boxes(query_boxes, 'query_boxes')
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)

    polys1, areas1 = [], np.zeros((N,), dtype=np.float64)
    for i in range(N):
        p = _rect_poly(boxes[i,0], boxes[i,1], boxes[i,2], boxes[i,3], boxes[i,4])
        polys1.append(p)
        areas1[i] = p.area if p.is_valid else 0.0

    polys2, areas2 = [], np.zeros((K,), dtype=np.float64)
    for j in range(K):
        p = _rect_poly(query_boxes[j,0], query_boxes[j,1], query_boxes[j,2], query_boxes[j,3], query_boxes[j,4])
        polys2.append(p)
        areas2[j] = p.area if p.is_valid else 0.0

    for i in range(N):
        a1 = areas1[i]
        if a1 <= 0:
            continue
        p1 = polys1[i]
        for j in range(K):
            a2 = areas2[j]
            if a2 <= 0:
                continue
            inter = p1.intersection(polys2[j]).area
            if inter <= 0:
                continue
            if criterion == 0:
                overlaps[i, j] = inter / a1
            elif criterion == 1:
                overlaps[i, j] = inter / a2
            elif criterion == -1:
                union = a1 + a2 - inter
                overlaps[i, j] = inter / union if union > 0 else 0.0
            else:
                overlaps[i, j] = inter
    return overlaps
