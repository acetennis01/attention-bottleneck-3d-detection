"""Render a Simple-BEV-style temporal comparison on KITTI Raw data.

The script runs the matched LiDAR-only and MBT checkpoints frame-by-frame on
a synced KITTI Raw drive, then writes a four-panel BEV comparison above the
front camera image.  The reference panel displays the full 360-degree raw
scan, while both models receive only the camera-visible subset matching KITTI
``velodyne_reduced``.  It intentionally performs detection only; no temporal
smoothing or tracking is applied to the predictions.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree

import cv2
import numpy as np
import torch
from mmengine.dataset import Compose, pseudo_collate

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.structures import get_box_type


CLASS_NAMES = ('Pedestrian', 'Cyclist', 'Car')
CLASS_COLORS = {
    0: (74, 190, 86),      # green, BGR
    1: (36, 164, 255),     # orange
    2: (232, 102, 45),     # blue
}
POINT_CLOUD_RANGE = (0.0, -39.68, 69.12, 39.68)
FULL_SCAN_RANGE = (-70.0, -70.0, 70.0, 70.0)
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@dataclass(frozen=True)
class DetectionSet:
    boxes: np.ndarray
    corners: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--drive-root',
        default=(
            'data/kitti_raw/2011_09_26/'
            '2011_09_26_drive_0005_sync'))
    parser.add_argument(
        '--calib-root', default='data/kitti_raw/2011_09_26')
    parser.add_argument(
        '--lidar-config',
        default='projects/myfusion/configs/pointpillars_lidar_control_eb48.py')
    parser.add_argument(
        '--lidar-checkpoint',
        default=(
            'work_dirs/pointpillars_lidar_control_eb48_stable_momentum_seed0/'
            'best_Kitti metric_pred_instances_3d_KITTI_'
            'Overall_3D_AP40_moderate_epoch_28.pth'))
    parser.add_argument(
        '--mbt-config',
        default='projects/myfusion/configs/my_fusion_mbt_bev_eb48.py')
    parser.add_argument(
        '--mbt-checkpoint',
        default=(
            'work_dirs/mbt_bev_eb48_stable_momentum_seed0/'
            'best_Kitti metric_pred_instances_3d_KITTI_'
            'Overall_3D_AP40_moderate_epoch_58.pth'))
    parser.add_argument(
        '--out-dir',
        default='work_dirs/temporal_visualization_drive_0005')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.30)
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument(
        '--max-frames', type=int, default=0,
        help='Process at most this many frames; 0 means the full drive.')
    return parser.parse_args()


def read_calibration_file(path: Path) -> Dict[str, np.ndarray]:
    values: Dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ':' not in line:
            continue
        key, raw_value = line.split(':', 1)
        try:
            values[key] = np.asarray(
                [float(value) for value in raw_value.split()],
                dtype=np.float64)
        except ValueError:
            continue
    return values


def build_lidar_to_image(calib_root: Path) -> np.ndarray:
    camera = read_calibration_file(calib_root / 'calib_cam_to_cam.txt')
    velo = read_calibration_file(calib_root / 'calib_velo_to_cam.txt')

    projection = camera['P_rect_02'].reshape(3, 4)
    rectification = np.eye(4, dtype=np.float64)
    rectification[:3, :3] = camera['R_rect_00'].reshape(3, 3)
    velo_to_camera = np.eye(4, dtype=np.float64)
    velo_to_camera[:3, :3] = velo['R'].reshape(3, 3)
    velo_to_camera[:3, 3] = velo['T']
    return projection @ rectification @ velo_to_camera


def boxes_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert LiDAR boxes (x, y, z, length, width, height, yaw) to corners."""
    if len(boxes) == 0:
        return np.empty((0, 8, 3), dtype=np.float32)
    all_corners = []
    for x, y, z, length, width, height, yaw in boxes[:, :7]:
        local = np.asarray([
            [-length / 2, width / 2, 0],
            [-length / 2, -width / 2, 0],
            [length / 2, -width / 2, 0],
            [length / 2, width / 2, 0],
            [-length / 2, width / 2, height],
            [-length / 2, -width / 2, height],
            [length / 2, -width / 2, height],
            [length / 2, width / 2, height],
        ], dtype=np.float32)
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.asarray(
            [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]],
            dtype=np.float32)
        all_corners.append(local @ rotation.T + np.asarray([x, y, z]))
    return np.stack(all_corners).astype(np.float32)


def parse_tracklets(path: Path) -> Dict[int, DetectionSet]:
    """Parse KITTI Raw tracklets into per-frame LiDAR boxes."""
    per_frame: Dict[int, List[Tuple[np.ndarray, int]]] = {}
    tracklets = ElementTree.parse(path).getroot().find('tracklets')
    if tracklets is None:
        raise ValueError(f'No tracklets element found in {path}')

    class_to_label = {name: index for index, name in enumerate(CLASS_NAMES)}
    for item in tracklets.findall('item'):
        object_type = item.findtext('objectType')
        if object_type not in class_to_label:
            continue
        height = float(item.findtext('h'))
        width = float(item.findtext('w'))
        length = float(item.findtext('l'))
        first_frame = int(item.findtext('first_frame'))
        poses = item.find('poses')
        if poses is None:
            continue
        pose_items = poses.findall('item')
        for offset, pose in enumerate(pose_items):
            truncation = int(pose.findtext('truncation', default='99'))
            if truncation not in (0, 1):
                continue
            box = np.asarray([
                float(pose.findtext('tx')),
                float(pose.findtext('ty')),
                float(pose.findtext('tz')),
                length,
                width,
                height,
                float(pose.findtext('rz')),
            ], dtype=np.float32)
            per_frame.setdefault(first_frame + offset, []).append(
                (box, class_to_label[object_type]))

    result: Dict[int, DetectionSet] = {}
    for frame_index, entries in per_frame.items():
        boxes = np.stack([entry[0] for entry in entries])
        labels = np.asarray([entry[1] for entry in entries], dtype=np.int64)
        result[frame_index] = DetectionSet(
            boxes=boxes,
            corners=boxes_to_corners(boxes),
            scores=np.ones(len(entries), dtype=np.float32),
            labels=labels,
        )
    return result


def empty_detections() -> DetectionSet:
    return DetectionSet(
        boxes=np.empty((0, 7), dtype=np.float32),
        corners=np.empty((0, 8, 3), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
        labels=np.empty((0,), dtype=np.int64),
    )


def unpack_predictions(result, score_threshold: float) -> DetectionSet:
    prediction = result.pred_instances_3d
    scores = prediction.scores_3d.detach().cpu().numpy()
    keep = scores >= score_threshold
    boxes = prediction.bboxes_3d.tensor.detach().cpu().numpy()[keep, :7]
    corners = prediction.bboxes_3d.corners.detach().cpu().numpy()[keep]
    labels = prediction.labels_3d.detach().cpu().numpy()[keep]
    return DetectionSet(
        boxes=boxes.astype(np.float32),
        corners=corners.astype(np.float32),
        scores=scores[keep].astype(np.float32),
        labels=labels.astype(np.int64),
    )


def build_multimodal_pipeline(model) -> Tuple[Compose, type, object]:
    dataset = model.cfg.test_dataloader.dataset
    pipeline_config = deepcopy(dataset.pipeline)
    for transform in pipeline_config:
        if transform.get('type') == 'LoadPointsFromFile':
            transform['type'] = 'LoadPointsFromDict'
    pipeline = Compose(pipeline_config)
    box_type_3d, box_mode_3d = get_box_type(dataset.box_type_3d)
    return pipeline, box_type_3d, box_mode_3d


def inference_multimodal(
        model, pipeline: Compose, box_type_3d, box_mode_3d,
        points: np.ndarray, image_path: Path):
    data = pipeline(dict(
        points=points,
        img_path=str(image_path),
        timestamp=1,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
    ))
    with torch.no_grad():
        return model.test_step(pseudo_collate([data]))[0]


def bev_coordinates(
        points_xy: np.ndarray, size: int,
        point_cloud_range: Tuple[float, float, float, float] =
        POINT_CLOUD_RANGE) -> np.ndarray:
    x_min, y_min, x_max, y_max = point_cloud_range
    u = (y_max - points_xy[:, 1]) / (y_max - y_min) * (size - 1)
    v = (x_max - points_xy[:, 0]) / (x_max - x_min) * (size - 1)
    return np.stack((u, v), axis=1).round().astype(np.int32)


def box_footprint(box: np.ndarray) -> np.ndarray:
    x, y, _, length, width, _, yaw = box[:7]
    local = np.asarray([
        [length / 2, width / 2],
        [-length / 2, width / 2],
        [-length / 2, -width / 2],
        [length / 2, -width / 2],
    ], dtype=np.float32)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.asarray([x, y])


def draw_bev_grid(panel: np.ndarray) -> None:
    size = panel.shape[0]
    for x in np.arange(0, 70, 10):
        endpoints = bev_coordinates(
            np.asarray([[x, POINT_CLOUD_RANGE[1]],
                        [x, POINT_CLOUD_RANGE[3]]]), size)
        cv2.line(panel, tuple(endpoints[0]), tuple(endpoints[1]),
                 (220, 220, 220), 1, cv2.LINE_AA)
    for y in np.arange(-30, 40, 10):
        endpoints = bev_coordinates(
            np.asarray([[POINT_CLOUD_RANGE[0], y],
                        [POINT_CLOUD_RANGE[2], y]]), size)
        cv2.line(panel, tuple(endpoints[0]), tuple(endpoints[1]),
                 (220, 220, 220), 1, cv2.LINE_AA)


def draw_title(panel: np.ndarray, title: str, dark: bool = False) -> None:
    color = (245, 245, 245) if dark else (35, 35, 35)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 38),
                  (20, 20, 20) if dark else (250, 250, 250), -1)
    cv2.putText(panel, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2, cv2.LINE_AA)


def render_lidar_panel(
        raw_points: np.ndarray, inference_points: np.ndarray,
        size: int) -> np.ndarray:
    """Render 360-degree raw LiDAR and highlight the model-input subset."""
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    x_min, y_min, x_max, y_max = FULL_SCAN_RANGE
    mask = ((raw_points[:, 0] >= x_min) & (raw_points[:, 0] <= x_max) &
            (raw_points[:, 1] >= y_min) & (raw_points[:, 1] <= y_max))
    visible = raw_points[mask]
    pixels = bev_coordinates(visible[:, :2], size, FULL_SCAN_RANGE)
    intensity = np.clip(visible[:, 3] * 255, 55, 255).astype(np.uint8)
    panel[pixels[:, 1], pixels[:, 0]] = np.stack(
        (intensity, intensity, intensity), axis=1)

    input_mask = (
        (inference_points[:, 0] >= x_min) &
        (inference_points[:, 0] <= x_max) &
        (inference_points[:, 1] >= y_min) &
        (inference_points[:, 1] <= y_max))
    input_pixels = bev_coordinates(
        inference_points[input_mask, :2], size, FULL_SCAN_RANGE)
    panel[input_pixels[:, 1], input_pixels[:, 0]] = (64, 185, 255)

    ego = bev_coordinates(
        np.asarray([[0.0, 0.0]], dtype=np.float32), size,
        FULL_SCAN_RANGE)[0]
    forward = bev_coordinates(
        np.asarray([[8.0, 0.0]], dtype=np.float32), size,
        FULL_SCAN_RANGE)[0]
    cv2.circle(panel, tuple(ego), 5, (80, 220, 80), -1, cv2.LINE_AA)
    cv2.arrowedLine(
        panel, tuple(ego), tuple(forward), (80, 220, 80), 2,
        cv2.LINE_AA, tipLength=0.25)
    draw_title(panel, 'LiDAR 360 | orange: model input', dark=True)
    return panel


def render_detection_panel(
        points: np.ndarray, detections: DetectionSet, size: int,
        title: str) -> np.ndarray:
    panel = np.full((size, size, 3), 247, dtype=np.uint8)
    draw_bev_grid(panel)
    x_min, y_min, x_max, y_max = POINT_CLOUD_RANGE
    mask = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
    sampled = points[mask][::3]
    pixels = bev_coordinates(sampled[:, :2], size)
    panel[pixels[:, 1], pixels[:, 0]] = (198, 198, 198)

    overlay = panel.copy()
    for box, label in zip(detections.boxes, detections.labels):
        polygon = bev_coordinates(box_footprint(box), size)
        color = CLASS_COLORS.get(int(label), (120, 120, 120))
        cv2.fillConvexPoly(overlay, polygon, color, cv2.LINE_AA)
    panel = cv2.addWeighted(overlay, 0.58, panel, 0.42, 0)

    for box, label in zip(detections.boxes, detections.labels):
        polygon = bev_coordinates(box_footprint(box), size)
        color = CLASS_COLORS.get(int(label), (120, 120, 120))
        cv2.polylines(panel, [polygon], True, color, 2, cv2.LINE_AA)
        center = bev_coordinates(box[None, :2], size)[0]
        heading_xy = box[:2] + np.asarray(
            [np.cos(box[6]), np.sin(box[6])]) * box[3] / 2
        heading = bev_coordinates(heading_xy[None], size)[0]
        cv2.line(panel, tuple(center), tuple(heading), color, 2, cv2.LINE_AA)

    ego = bev_coordinates(np.asarray([[0.8, 0.0]]), size)[0]
    cv2.circle(panel, tuple(ego), 6, (40, 40, 40), -1, cv2.LINE_AA)
    draw_title(panel, f'{title}  ({len(detections.boxes)})')
    return panel


def project_corners(corners: np.ndarray,
                    lidar_to_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    homogeneous = np.concatenate(
        (corners, np.ones((*corners.shape[:-1], 1), dtype=corners.dtype)),
        axis=-1)
    projected = homogeneous @ lidar_to_image.T
    valid = projected[..., 2] > 0.1
    pixels = projected[..., :2] / np.maximum(projected[..., 2:3], 1e-6)
    return pixels, valid


def draw_camera_boxes(
        image: np.ndarray, detections: DetectionSet,
        lidar_to_image: np.ndarray) -> np.ndarray:
    output = image.copy()
    pixels, valid = project_corners(detections.corners, lidar_to_image)
    height, width = output.shape[:2]
    for box_pixels, box_valid, label, score in zip(
            pixels, valid, detections.labels, detections.scores):
        color = CLASS_COLORS.get(int(label), (120, 120, 120))
        for start, end in BOX_EDGES:
            if not (box_valid[start] and box_valid[end]):
                continue
            first = tuple(np.round(box_pixels[start]).astype(int))
            second = tuple(np.round(box_pixels[end]).astype(int))
            cv2.line(output, first, second, color, 2, cv2.LINE_AA)
        visible = box_pixels[box_valid]
        if len(visible):
            x = int(np.clip(visible[:, 0].min(), 0, width - 1))
            y = int(np.clip(visible[:, 1].min(), 20, height - 1))
            label_text = f'{CLASS_NAMES[int(label)]} {score:.2f}'
            cv2.putText(output, label_text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2,
                        cv2.LINE_AA)
    return output


def fit_camera_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = max(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_LINEAR)
    y_start = max(0, (resized.shape[0] - height) // 2)
    x_start = max(0, (resized.shape[1] - width) // 2)
    return resized[y_start:y_start + height, x_start:x_start + width]


def compose_frame(
        drive_name: str, frame_index: int, raw_points: np.ndarray,
        inference_points: np.ndarray, camera: np.ndarray,
        lidar_predictions: DetectionSet, mbt_predictions: DetectionSet,
        ground_truth: DetectionSet, lidar_to_image: np.ndarray,
        score_threshold: float) -> np.ndarray:
    width, header_height, panel_size, camera_height = 1600, 64, 400, 480
    canvas = np.full(
        (header_height + panel_size + camera_height, width, 3),
        255, dtype=np.uint8)
    display_name = drive_name.replace('2011_09_26_drive_', 'drive ')
    title = (
        f'KITTI Raw {display_name}  |  '
        f'frame {frame_index:010d}  |  threshold {score_threshold:.2f}')
    cv2.putText(canvas, title, (32, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.92, (25, 25, 25), 2, cv2.LINE_AA)

    panels = [
        render_lidar_panel(raw_points, inference_points, panel_size),
        render_detection_panel(
            inference_points, lidar_predictions, panel_size, 'LiDAR-only'),
        render_detection_panel(
            inference_points, mbt_predictions, panel_size, 'MBT fusion'),
        render_detection_panel(
            inference_points, ground_truth, panel_size, 'Ground truth'),
    ]
    for index, panel in enumerate(panels):
        x_start = index * panel_size
        canvas[header_height:header_height + panel_size,
               x_start:x_start + panel_size] = panel

    camera_overlay = draw_camera_boxes(
        camera, mbt_predictions, lidar_to_image)
    camera_panel = fit_camera_image(camera_overlay, width, camera_height)
    canvas[header_height + panel_size:] = camera_panel

    legend_x = 1115
    cv2.rectangle(canvas, (legend_x - 18, 12), (1585, 54),
                  (255, 255, 255), -1)
    for index, (label, name) in enumerate(zip((2, 0, 1),
                                               ('Car', 'Pedestrian', 'Cyclist'))):
        x = legend_x + index * 150
        cv2.rectangle(canvas, (x, 25), (x + 18, 43),
                      CLASS_COLORS[label], -1)
        cv2.putText(canvas, name, (x + 25, 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (35, 35, 35), 1,
                    cv2.LINE_AA)
    return canvas


def collect_frames(drive_root: Path, stride: int,
                   max_frames: int) -> List[Tuple[int, Path, Path]]:
    image_dir = drive_root / 'image_02' / 'data'
    point_dir = drive_root / 'velodyne_points' / 'data'
    frames = []
    for image_path in sorted(image_dir.glob('*.png'))[::stride]:
        frame_index = int(image_path.stem)
        point_path = point_dir / f'{image_path.stem}.bin'
        if point_path.is_file():
            frames.append((frame_index, image_path, point_path))
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        raise FileNotFoundError(f'No synchronized frames found in {drive_root}')
    return frames


def reduce_points_to_camera(
        points: np.ndarray, lidar_to_image: np.ndarray,
        image_shape: Tuple[int, int]) -> np.ndarray:
    """Match KITTI ``velodyne_reduced`` by keeping camera-visible points."""
    homogeneous = np.concatenate(
        (points[:, :3], np.ones((len(points), 1), dtype=points.dtype)),
        axis=1)
    projected = homogeneous @ lidar_to_image.T
    depth = projected[:, 2]
    pixels = projected[:, :2] / np.maximum(depth[:, None], 1e-6)
    height, width = image_shape
    keep = ((depth > 0) &
            (pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
            (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    return points[keep]


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError('--frame-stride must be at least 1')
    if not (0 <= args.score_thr <= 1):
        raise ValueError('--score-thr must be between 0 and 1')

    drive_root = Path(args.drive_root)
    drive_name = drive_root.name.removesuffix('_sync')
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = collect_frames(drive_root, args.frame_stride, args.max_frames)
    ground_truth = parse_tracklets(drive_root / 'tracklet_labels.xml')
    lidar_to_image = build_lidar_to_image(Path(args.calib_root))

    print('Loading LiDAR-only checkpoint...')
    lidar_model = init_model(
        args.lidar_config, args.lidar_checkpoint, device=args.device)
    print('Loading MBT checkpoint...')
    mbt_model = init_model(
        args.mbt_config, args.mbt_checkpoint, device=args.device)
    mbt_pipeline, box_type_3d, box_mode_3d = build_multimodal_pipeline(
        mbt_model)

    video_path = output_dir / f'{drive_name}_full360_lidar_vs_mbt.mp4'
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
        (1600, 944))
    if not writer.isOpened():
        raise RuntimeError('OpenCV could not initialize the mp4v video writer')

    try:
        for position, (frame_index, image_path, point_path) in enumerate(frames):
            raw_points = np.fromfile(
                point_path, dtype=np.float32).reshape(-1, 4)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f'Could not read {image_path}')
            points = reduce_points_to_camera(
                raw_points, lidar_to_image, image.shape[:2])

            lidar_result, _ = inference_detector(lidar_model, points)
            mbt_result = inference_multimodal(
                mbt_model, mbt_pipeline, box_type_3d, box_mode_3d,
                points, image_path)
            lidar_predictions = unpack_predictions(
                lidar_result, args.score_thr)
            mbt_predictions = unpack_predictions(mbt_result, args.score_thr)
            gt = ground_truth.get(frame_index, empty_detections())
            rendered = compose_frame(
                drive_name, frame_index, raw_points, points, image,
                lidar_predictions, mbt_predictions, gt, lidar_to_image,
                args.score_thr)
            writer.write(rendered)
            if position == 0:
                cv2.imwrite(str(output_dir / 'preview.png'), rendered)
            if (position + 1) % 10 == 0 or position + 1 == len(frames):
                print(f'Processed {position + 1}/{len(frames)} frames')
    finally:
        writer.release()

    manifest = {
        'drive_root': str(drive_root),
        'drive_name': drive_name,
        'frame_count': len(frames),
        'first_frame': frames[0][0],
        'last_frame': frames[-1][0],
        'fps': args.fps,
        'frame_stride': args.frame_stride,
        'score_threshold': args.score_thr,
        'lidar_reference_panel': 'full 360-degree raw scan',
        'model_lidar_input': 'camera-FOV-reduced points',
        'lidar_config': args.lidar_config,
        'lidar_checkpoint': args.lidar_checkpoint,
        'mbt_config': args.mbt_config,
        'mbt_checkpoint': args.mbt_checkpoint,
        'video': str(video_path),
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {video_path}')


if __name__ == '__main__':
    main()
