"""Render class-specific camera/BEV attention maps for camera-focused MBT.

The visualization is a head-averaged, last-layer attention rollout from a
selected LiDAR detection token through the shared bottleneck tokens to the
geometrically aligned camera tokens. Maps are normalized independently for
display and must not be compared by color magnitude across examples.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import mmengine
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
import numpy as np
import torch

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS


CLASS_NAMES = ('Pedestrian', 'Cyclist', 'Car')
CLASS_COLORS = {
    0: (74, 190, 86),
    1: (36, 164, 255),
    2: (232, 102, 45),
}


@dataclass(frozen=True)
class Example:
    dataset_index: int
    sample_idx: int
    label: int
    bbox_2d: Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default=(
            'projects/myfusion/configs/'
            'my_fusion_mbt_bev_camera_focused_eb48.py'))
    parser.add_argument(
        '--checkpoint',
        default=(
            'work_dirs/mbt_bev_camera_focused_eb48_seed0/'
            'best_Kitti metric_pred_instances_3d_KITTI_'
            'Overall_3D_AP40_moderate_epoch_20.pth'))
    parser.add_argument(
        '--info-file', default='data/kitti/kitti_infos_val.pkl')
    parser.add_argument(
        '--predictions-file',
        default=(
            'work_dirs/mbt_bev_camera_focused_best_eval/predictions/'
            'pred_instances_3d.pkl'))
    parser.add_argument(
        '--output',
        default=(
            'projects/myfusion/docs/'
            'attention_maps_camera_bev.png'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.30)
    parser.add_argument('--candidate-limit', type=int, default=80)
    parser.add_argument(
        '--exclude-samples', nargs='*', type=int,
        default=(),
        help='Sample IDs excluded from automatic example selection.')
    parser.add_argument(
        '--pedestrian-samples', nargs='*', type=int, default=(6804,))
    parser.add_argument(
        '--cyclist-samples', nargs='*', type=int, default=(2206,))
    parser.add_argument(
        '--car-samples', nargs='*', type=int, default=(767,))
    return parser.parse_args()


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-12)


def find_candidates(
        data_list: Sequence[dict],
        persisted_predictions: Sequence[dict],
        excluded_samples: Sequence[int]) -> Dict[int, List[Example]]:
    """Find reproducible moderate-difficulty true positives per class."""
    result: Dict[int, List[Example]] = {}
    excluded = set(excluded_samples)
    for label in range(len(CLASS_NAMES)):
        candidates = []
        for dataset_index, info in enumerate(data_list):
            sample_idx = int(info['sample_idx'])
            if sample_idx in excluded:
                continue
            prediction = persisted_predictions[dataset_index]
            predicted_names = np.asarray(prediction['name'])
            predicted_boxes = np.asarray(prediction['bbox'])
            predicted_scores = np.asarray(prediction['score'])
            class_mask = predicted_names == CLASS_NAMES[label]
            for instance in info.get('instances', []):
                if (instance.get('bbox_label_3d') != label or
                        instance.get('difficulty') != 1):
                    continue
                depth = float(instance.get('depth', 0.0))
                points = int(instance.get('num_lidar_pts', 0))
                if not 7.0 <= depth <= 45.0 or points < 5:
                    continue
                if not class_mask.any():
                    continue
                target_box = tuple(
                    float(value) for value in instance['bbox'])
                overlaps = np.asarray([
                    bbox_iou(target_box, box)
                    for box in predicted_boxes[class_mask]
                ])
                best_index = int(overlaps.argmax())
                overlap = float(overlaps[best_index])
                score = float(predicted_scores[class_mask][best_index])
                if overlap < 0.50 or score < 0.30:
                    continue
                candidates.append((
                    score,
                    overlap,
                    dataset_index,
                    sample_idx,
                    target_box,
                ))
        candidates.sort(reverse=True)
        result[label] = [
            Example(
                dataset_index=dataset_index,
                sample_idx=sample_idx,
                label=label,
                bbox_2d=bbox_2d,
            )
            for _, _, dataset_index, sample_idx, bbox_2d in candidates
        ]
    return result


def run_model_with_attention(model, dataset, dataset_index: int):
    packed = pseudo_collate([dataset[dataset_index]])
    batch = model.data_preprocessor(packed, training=False)
    inputs = batch['inputs']
    samples = batch['data_samples']
    with torch.no_grad():
        image_features = model.extract_img_feat(inputs['imgs'])
        lidar_features = model.extract_pts_feat(inputs['voxels'], 1)
        metadata = model._projection_metadata(samples, inputs['imgs'])
        fusion = model.fusion_module(
            image_features,
            lidar_features,
            return_attention=True,
            **metadata,
        )
        predictions = model.bbox_head.predict(
            (fusion['bev_features'], ), samples)
    return fusion, predictions[0], samples[0]


def projected_bbox(corners: np.ndarray, lidar2img: np.ndarray):
    points = np.concatenate(
        (corners, np.ones((corners.shape[0], 1), dtype=np.float32)), axis=1)
    projected = points @ lidar2img.T
    if np.count_nonzero(projected[:, 2] > 1e-5) < 4:
        return None
    u = projected[:, 0] / np.maximum(projected[:, 2], 1e-5)
    v = projected[:, 1] / np.maximum(projected[:, 2], 1e-5)
    return (float(u.min()), float(v.min()), float(u.max()), float(v.max()))


def select_detection(
        prediction, label: int, target_bbox: Sequence[float],
        lidar2img: np.ndarray):
    scores = prediction.scores_3d.detach().cpu()
    labels = prediction.labels_3d.detach().cpu()
    boxes = prediction.bboxes_3d.tensor.detach().cpu()
    indices = torch.nonzero(labels == label).flatten()
    if indices.numel() == 0:
        index = int(scores.argmax())
    else:
        corners = prediction.bboxes_3d.corners.detach().cpu().numpy()
        overlaps = []
        for candidate_index in indices:
            box = projected_bbox(corners[int(candidate_index)], lidar2img)
            overlaps.append(0.0 if box is None else bbox_iou(target_bbox, box))
        index = int(indices[int(np.argmax(overlaps))])
    return (
        boxes[index].numpy(),
        float(scores[index]),
        int(labels[index]),
        prediction.bboxes_3d.corners[index].detach().cpu().numpy(),
    )


def attention_rollout(
        fusion: Dict[str, torch.Tensor], box: np.ndarray,
        point_cloud_range: Sequence[float],
        grid_size: Tuple[int, int]) -> Tuple[np.ndarray, int, Dict[str, float]]:
    grid_h, grid_w = grid_size
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    grid_x = int(np.floor((box[0] - x_min) / (x_max - x_min) * grid_w))
    grid_y = int(np.floor((box[1] - y_min) / (y_max - y_min) * grid_h))
    grid_x = int(np.clip(grid_x, 0, grid_w - 1))
    grid_y = int(np.clip(grid_y, 0, grid_h - 1))
    query_index = grid_y * grid_w + grid_x

    attention = fusion['attention']
    # [heads, bottlenecks, camera cells] and [heads, lidar cells, bottlenecks]
    image_to_bottleneck = attention['image_bottleneck'][0, -1].mean(dim=0)
    bottleneck_to_lidar = attention['lidar_readout'][0].mean(dim=0)
    rollout = torch.matmul(
        bottleneck_to_lidar[query_index], image_to_bottleneck)
    valid = fusion['image_valid_mask'][0, :, 0]
    rollout = rollout * valid.to(rollout.dtype)
    rollout = rollout / rollout.sum().clamp_min(1e-12)
    values = rollout.detach().cpu().numpy().reshape(grid_h, grid_w)

    positive = values[values > 0]
    entropy = float(-(positive * np.log(positive + 1e-12)).sum())
    normalized_entropy = entropy / max(np.log(len(positive)), 1e-12)
    count = max(1, int(np.ceil(0.10 * len(positive))))
    top_ten_mass = float(np.sort(positive)[-count:].sum())
    statistics = dict(
        normalized_entropy=normalized_entropy,
        top_10_percent_mass=top_ten_mass,
        valid_cells=int(len(positive)),
    )
    return values, query_index, statistics


def robust_normalize(values: np.ndarray, mask: np.ndarray | None = None):
    if mask is None:
        selected = values[values > 0]
    else:
        selected = values[mask]
    if selected.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(selected, (5, 99))
    if high <= low:
        high = float(selected.max())
        low = float(selected.min())
    return np.clip((values - low) / max(high - low, 1e-12), 0, 1)


def camera_heatmap(
        attention: np.ndarray, height_weights: np.ndarray,
        lidar2img: np.ndarray, image_shape: Tuple[int, int],
        point_cloud_range: Sequence[float],
        height_anchors: Sequence[float]) -> np.ndarray:
    grid_h, grid_w = attention.shape
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    x = np.linspace(x_min, x_max, grid_w + 1)
    y = np.linspace(y_min, y_max, grid_h + 1)
    x = 0.5 * (x[:-1] + x[1:])
    y = 0.5 * (y[:-1] + y[1:])
    yy, xx = np.meshgrid(y, x, indexing='ij')

    heat = np.zeros(image_shape, dtype=np.float32)
    for height_index, height in enumerate(height_anchors):
        points = np.stack((
            xx,
            yy,
            np.full_like(xx, height),
            np.ones_like(xx),
        ), axis=-1).reshape(-1, 4)
        projected = points @ lidar2img.T
        depth = projected[:, 2]
        u = projected[:, 0] / np.maximum(depth, 1e-5)
        v = projected[:, 1] / np.maximum(depth, 1e-5)
        values = (
            attention * height_weights[height_index]).reshape(-1)
        valid = (
            (depth > 1e-5) & (u >= 0) & (u < image_shape[1]) &
            (v >= 0) & (v < image_shape[0]) & (values > 0))
        pixels_x = np.rint(u[valid]).astype(np.int32)
        pixels_y = np.rint(v[valid]).astype(np.int32)
        np.add.at(heat, (pixels_y, pixels_x), values[valid])
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=22, sigmaY=22)
    return robust_normalize(heat, heat > 0)


def object_attention_score(
        heat: np.ndarray,
        bbox: Sequence[float]) -> Dict[str, float]:
    """Score how clearly the rendered attention overlaps an object box."""
    height, width = heat.shape
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(np.floor(x1), 0, width - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, width))
    y1 = int(np.clip(np.floor(y1), 0, height - 1))
    y2 = int(np.clip(np.ceil(y2), y1 + 1, height))
    inside = heat[y1:y2, x1:x2]

    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    expanded_width = 1.6 * (x2 - x1)
    expanded_height = 1.6 * (y2 - y1)
    ex1 = int(np.clip(center_x - expanded_width / 2, 0, width - 1))
    ex2 = int(np.clip(center_x + expanded_width / 2, ex1 + 1, width))
    ey1 = int(np.clip(center_y - expanded_height / 2, 0, height - 1))
    ey2 = int(np.clip(center_y + expanded_height / 2, ey1 + 1, height))
    expanded = heat[ey1:ey2, ex1:ex2]

    peak = float(np.percentile(inside, 95))
    mean = float(inside.mean())
    context_mean = float(expanded.mean())
    score = 0.55 * peak + 0.30 * mean + 0.15 * context_mean
    return dict(
        selection_score=score,
        object_attention_peak=peak,
        object_attention_mean=mean,
        object_context_attention_mean=context_mean,
    )


def bev_query_attention_score(
        attention: np.ndarray, query_index: int) -> Dict[str, float]:
    """Measure source-attention mass near the selected LiDAR query cell."""
    grid_h, grid_w = attention.shape
    query_y, query_x = divmod(query_index, grid_w)
    y1, y2 = max(0, query_y - 1), min(grid_h, query_y + 2)
    x1, x2 = max(0, query_x - 1), min(grid_w, query_x + 2)
    local_mass = float(attention[y1:y2, x1:x2].sum())
    center_mass = float(attention[query_y, query_x])
    # Roughly 0.05 is uniform for a 3x3 patch over 169 visible cells.
    focus_score = min(local_mass / 0.20, 1.0)
    return dict(
        bev_query_center_mass=center_mass,
        bev_query_local_mass=local_mass,
        bev_query_focus_score=float(focus_score),
    )


def apply_heatmap(image: np.ndarray, heat: np.ndarray, alpha: float = 0.58):
    color = cv2.applyColorMap(
        np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_TURBO)
    opacity = (alpha * heat)[..., None]
    return np.uint8(
        image.astype(np.float32) * (1 - opacity) +
        color.astype(np.float32) * opacity)


def bev_coordinates(
        points_xy: np.ndarray, size: int,
        point_cloud_range: Sequence[float]) -> np.ndarray:
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    u = (y_max - points_xy[:, 1]) / (y_max - y_min) * (size - 1)
    v = (x_max - points_xy[:, 0]) / (x_max - x_min) * (size - 1)
    return np.stack((u, v), axis=1).round().astype(np.int32)


def render_bev(
        attention: np.ndarray, points: np.ndarray, prediction,
        selected_corners: np.ndarray, point_cloud_range: Sequence[float],
        score_threshold: float, size: int = 420) -> np.ndarray:
    panel_attention = np.flip(attention.T, axis=(0, 1))
    panel_attention = cv2.resize(
        robust_normalize(panel_attention), (size, size),
        interpolation=cv2.INTER_CUBIC)
    panel = np.full((size, size, 3), 245, dtype=np.uint8)
    panel = apply_heatmap(panel, panel_attention, alpha=0.66)

    mask = (
        (points[:, 0] >= point_cloud_range[0]) &
        (points[:, 0] <= point_cloud_range[3]) &
        (points[:, 1] >= point_cloud_range[1]) &
        (points[:, 1] <= point_cloud_range[4]))
    pixels = bev_coordinates(points[mask, :2], size, point_cloud_range)
    panel[pixels[:, 1], pixels[:, 0]] = (55, 55, 55)

    scores = prediction.scores_3d.detach().cpu().numpy()
    labels = prediction.labels_3d.detach().cpu().numpy()
    corners = prediction.bboxes_3d.corners.detach().cpu().numpy()
    for score, label, box_corners in zip(scores, labels, corners):
        if score < score_threshold:
            continue
        footprint = bev_coordinates(
            box_corners[:4, :2], size, point_cloud_range)
        cv2.polylines(
            panel, [footprint], True, CLASS_COLORS[int(label)],
            2, cv2.LINE_AA)
    target = bev_coordinates(
        selected_corners[:4, :2], size, point_cloud_range)
    cv2.polylines(panel, [target], True, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.polylines(panel, [target], True, (20, 20, 20), 2, cv2.LINE_AA)
    center = bev_coordinates(
        selected_corners[:4, :2].mean(axis=0, keepdims=True),
        size,
        point_cloud_range,
    )[0]
    cv2.circle(panel, tuple(center), 8, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(panel, tuple(center), 5, (255, 255, 255), -1, cv2.LINE_AA)
    return panel


def titled(panel: np.ndarray, title: str, subtitle: str = '') -> np.ndarray:
    header = np.full((60, panel.shape[1], 3), 250, dtype=np.uint8)
    cv2.putText(
        header, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (25, 25, 25), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(
            header, subtitle, (12, 49), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (70, 70, 70), 1, cv2.LINE_AA)
    return np.concatenate((header, panel), axis=0)


def resize_and_pad(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    info_bundle = mmengine.load(args.info_file)
    data_list = info_bundle['data_list']
    persisted_predictions = mmengine.load(args.predictions_file)
    point_cloud_range = tuple(
        float(value) for value in model.fusion_module.point_cloud_range.cpu())
    heights = tuple(
        float(value) for value in model.fusion_module.height_anchors.cpu())
    grid_size = model.fusion_module.token_grid_size

    candidate_groups = find_candidates(
        data_list, persisted_predictions, args.exclude_samples)
    requested_samples = (
        args.pedestrian_samples,
        args.cyclist_samples,
        args.car_samples,
    )
    for label, sample_ids in enumerate(requested_samples):
        if sample_ids:
            allowed = set(sample_ids)
            candidate_groups[label] = [
                example for example in candidate_groups[label]
                if example.sample_idx in allowed
            ]
    examples = []
    selection_details: Dict[int, Dict[str, float]] = {}
    used_samples = set()
    for label in range(len(CLASS_NAMES)):
        best = None
        for example in candidate_groups[label][:args.candidate_limit]:
            if example.sample_idx in used_samples:
                continue
            fusion, prediction, sample = run_model_with_attention(
                model, dataset, example.dataset_index)
            info = data_list[example.dataset_index]
            lidar2img = np.asarray(
                info['images']['CAM2']['lidar2img'], dtype=np.float32)
            box, score, predicted_label, _ = select_detection(
                prediction, example.label, example.bbox_2d, lidar2img)
            if predicted_label != label or score < args.score_thr:
                continue
            attention, query_index, _ = attention_rollout(
                fusion, box, point_cloud_range, grid_size)
            height_weights = (
                fusion['attention']['image_height_weights'][0]
                .detach().cpu().numpy())
            image_path = Path(sample.metainfo['img_path'])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            heat = camera_heatmap(
                attention,
                height_weights,
                lidar2img,
                image.shape[:2],
                point_cloud_range,
                heights,
            )
            details = object_attention_score(heat, example.bbox_2d)
            details.update(bev_query_attention_score(
                attention, query_index))
            x1, y1, x2, y2 = example.bbox_2d
            object_scale = min(
                np.sqrt(max(0.0, x2 - x1) * max(0.0, y2 - y1)) / 80.0,
                1.0,
            )
            figure_score = (
                0.35 * details['selection_score'] +
                0.30 * score +
                0.10 * object_scale +
                0.25 * details['bev_query_focus_score'])
            details['object_scale_score'] = float(object_scale)
            details['figure_selection_score'] = float(figure_score)
            rank = (figure_score, details['selection_score'], score)
            if best is None or rank > best[0]:
                best = (rank, example, details)
        if best is None:
            raise RuntimeError(
                f'No usable attention example found for {CLASS_NAMES[label]}')
        _, example, details = best
        examples.append(example)
        selection_details[label] = details
        used_samples.add(example.sample_idx)
        print(
            f'Selected {CLASS_NAMES[label]} sample {example.sample_idx:06d} '
            f'with attention-overlap score '
            f'{details["selection_score"]:.4f}')

    rows = []
    summaries = []

    for example in examples:
        fusion, prediction, sample = run_model_with_attention(
            model, dataset, example.dataset_index)
        info = data_list[example.dataset_index]
        lidar2img = np.asarray(
            info['images']['CAM2']['lidar2img'], dtype=np.float32)
        box, score, predicted_label, corners = select_detection(
            prediction, example.label, example.bbox_2d, lidar2img)
        attention, query_index, statistics = attention_rollout(
            fusion, box, point_cloud_range, grid_size)
        height_weights = (
            fusion['attention']['image_height_weights'][0]
            .detach().cpu().numpy())

        image_path = Path(sample.metainfo['img_path'])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image_heat = camera_heatmap(
            attention,
            height_weights,
            lidar2img,
            image.shape[:2],
            point_cloud_range,
            heights,
        )
        camera_panel = apply_heatmap(image, image_heat)
        x1, y1, x2, y2 = (round(value) for value in example.bbox_2d)
        cv2.rectangle(
            camera_panel, (x1, y1), (x2, y2), (255, 255, 255),
            4, cv2.LINE_AA)
        cv2.rectangle(
            camera_panel, (x1, y1), (x2, y2), (20, 20, 20),
            2, cv2.LINE_AA)

        point_path = (
            image_path.parent.parent / 'velodyne_reduced' /
            f'{example.sample_idx:06d}.bin')
        points = np.fromfile(point_path, dtype=np.float32).reshape(-1, 4)
        bev_panel = render_bev(
            attention,
            points,
            prediction,
            corners,
            point_cloud_range,
            args.score_thr,
        )
        camera_panel = resize_and_pad(camera_panel, 840, 420)
        class_name = CLASS_NAMES[example.label]
        camera_panel = titled(
            camera_panel,
            f'{class_name} -- camera attention',
            f'KITTI {example.sample_idx:06d}; white = selected object')
        bev_panel = titled(
            bev_panel,
            'BEV attention and predictions',
            f'confidence={score:.3f}; white = selected detection')
        row = np.concatenate((camera_panel, bev_panel), axis=1)
        rows.append(row)
        summaries.append(dict(
            class_name=class_name,
            sample_idx=example.sample_idx,
            dataset_index=example.dataset_index,
            predicted_class=CLASS_NAMES[predicted_label],
            prediction_score=score,
            query_token_index=query_index,
            **selection_details[example.label],
            **statistics,
        ))

    separator = np.full((14, rows[0].shape[1], 3), 255, dtype=np.uint8)
    figure_parts = []
    for index, row in enumerate(rows):
        if index:
            figure_parts.append(separator)
        figure_parts.append(row)
    figure = np.concatenate(figure_parts, axis=0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), figure):
        raise RuntimeError(f'Could not write {output}')
    summary_path = output.with_name('attention_map_summary.json')
    summary_path.write_text(json.dumps(summaries, indent=2) + '\n')
    print(f'Wrote {output}')
    print(f'Wrote {summary_path}')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
