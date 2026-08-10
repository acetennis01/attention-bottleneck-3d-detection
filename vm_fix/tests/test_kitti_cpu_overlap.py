"""Regression tests for the KITTI rotated-overlap CPU backend."""

import numpy as np

from mmdet3d.evaluation.functional.kitti_utils.eval import (
    bev_box_overlap, d3_box_overlap)
from mmdet3d.evaluation.functional.kitti_utils.rotate_iou_cpu import (
    rotate_iou_cpu_eval)
from mmdet3d.evaluation.functional.kitti_utils.rotate_iou_mmcv import (
    rotate_iou_mmcv_eval)


def test_cpu_overlap_matches_official_criterion_semantics():
    first = np.array([[0.0, 0.0, 4.0, 2.0, 0.0]])
    same = first.copy()
    shifted = np.array([[2.0, 0.0, 4.0, 2.0, 0.0]])

    np.testing.assert_allclose(rotate_iou_cpu_eval(first, same, -1), [[1.0]])
    np.testing.assert_allclose(rotate_iou_cpu_eval(first, same, 0), [[1.0]])
    np.testing.assert_allclose(rotate_iou_cpu_eval(first, same, 1), [[1.0]])
    np.testing.assert_allclose(rotate_iou_cpu_eval(first, same, 2), [[8.0]])

    np.testing.assert_allclose(
        rotate_iou_cpu_eval(first, shifted, -1), [[1.0 / 3.0]])
    np.testing.assert_allclose(
        rotate_iou_cpu_eval(first, shifted, 0), [[0.5]])
    np.testing.assert_allclose(
        rotate_iou_cpu_eval(first, shifted, 1), [[0.5]])
    np.testing.assert_allclose(
        rotate_iou_cpu_eval(first, shifted, 2), [[4.0]])


def test_identical_camera_boxes_have_unit_bev_and_3d_iou():
    # Camera box layout: x, y(bottom), z, length, height, width, yaw.
    box = np.array([[1.0, 1.5, 12.0, 4.0, 1.5, 1.8, 0.3]])
    np.testing.assert_allclose(bev_box_overlap(box[:, [0, 2, 3, 5, 6]],
                                               box[:, [0, 2, 3, 5, 6]]),
                               [[1.0]])
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
