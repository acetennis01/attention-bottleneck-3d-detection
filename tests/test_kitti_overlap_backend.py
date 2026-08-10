"""Regression tests for the KITTI evaluator compatibility backend."""

import numpy as np

from mmdet3d.evaluation.functional.kitti_utils.eval import (
    bev_box_overlap, d3_box_overlap)
from mmdet3d.evaluation.functional.kitti_utils.rotate_iou_cpu import (
    rotate_iou_cpu_eval)
from mmdet3d.evaluation.functional.kitti_utils.rotate_iou_mmcv import (
    rotate_iou_mmcv_eval)


def test_raw_intersection_criterion_is_not_iou():
    first = np.array([[0.0, 0.0, 4.0, 2.0, 0.0]])
    shifted = np.array([[2.0, 0.0, 4.0, 2.0, 0.0]])
    np.testing.assert_allclose(
        rotate_iou_cpu_eval(first, shifted, -1), [[1.0 / 3.0]])
    np.testing.assert_allclose(
        rotate_iou_cpu_eval(first, shifted, 2), [[4.0]])


def test_identical_camera_boxes_have_unit_bev_and_3d_iou():
    box = np.array([[1.0, 1.5, 12.0, 4.0, 1.5, 1.8, 0.3]])
    planar = box[:, [0, 2, 3, 5, 6]]
    np.testing.assert_allclose(bev_box_overlap(planar, planar), [[1.0]])
    np.testing.assert_allclose(d3_box_overlap(box, box), [[1.0]])


def test_mmcv_backend_matches_cpu_reference_for_all_criteria():
    boxes = np.array([
        [0.0, 0.0, 4.0, 2.0, 0.35],
        [3.0, -1.0, 1.5, 3.2, -0.7],
        [10.0, 4.0, 2.0, 2.0, 1.2],
    ])
    queries = np.array([
        [0.6, 0.4, 3.5, 1.8, -0.2],
        [3.5, -0.4, 2.0, 2.8, 0.9],
    ])
    for criterion in (-1, 0, 1, 2):
        np.testing.assert_allclose(
            rotate_iou_mmcv_eval(boxes, queries, criterion),
            rotate_iou_cpu_eval(boxes, queries, criterion),
            rtol=2e-5,
            atol=2e-6,
        )
