"""Render nuScenes camera, BEV detection, and MBT-attention visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import cv2
import mmengine
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
import numpy as np
import torch

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS


CLASS_NAMES = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
CLASS_COLORS = (
    (230, 126, 34), (52, 152, 219), (155, 89, 182), (46, 204, 113),
    (127, 140, 141), (241, 196, 15), (231, 76, 60), (26, 188, 156),
    (22, 160, 240), (149, 165, 166))
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default='projects/myfusion/configs/my_fusion_mbt_bev_nuscenes.py')
    parser.add_argument(
        '--checkpoint',
        default=(
            'work_dirs/mbt_bev_nuscenes_fixed/'
            'best_NuScenes metric_pred_instances_3d_NuScenes_NDS_epoch_9.pth'))
    parser.add_argument(
        '--info-file', default='data/nuscenes_full/nuscenes_infos_val.pkl')
    parser.add_argument('--nuscenes-root', default='data/nuscenes_full')
    parser.add_argument(
        '--results-json',
        default=(
            'work_dirs/nuscenes_analysis/correct/results/'
            'pred_instances_3d/results_nusc.json'))
    parser.add_argument(
        '--output-dir', default='work_dirs/nuscenes_analysis/visualizations')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--static-count', type=int, default=6)
    parser.add_argument('--video-frames', type=int, default=18)
    parser.add_argument('--video-fps', type=float, default=2.0)
    return parser.parse_args()


def resize_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = max(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA)
    x = max(0, (resized.shape[1] - width) // 2)
    y = max(0, (resized.shape[0] - height) // 2)
    return resized[y:y + height, x:x + width].copy()


def put_label(image: np.ndarray, text: str, origin=(12, 28), scale=0.65,
              color=(255, 255, 255)) -> None:
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, 2, cv2.LINE_AA)


def compose_lidar2img(cam2img: np.ndarray, lidar2cam: np.ndarray) -> np.ndarray:
    if cam2img.shape == (3, 3):
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = cam2img
    else:
        intrinsic = cam2img.astype(np.float32)
    return intrinsic @ lidar2cam.astype(np.float32)


def draw_projected_boxes(
        image: np.ndarray, corners: np.ndarray, scores: np.ndarray,
        labels: np.ndarray, lidar2img: np.ndarray, score_thr: float) -> None:
    height, width = image.shape[:2]
    for box, score, label in zip(corners, scores, labels):
        if score < score_thr:
            continue
        homogeneous = np.concatenate(
            (box, np.ones((8, 1), dtype=np.float32)), axis=1)
        projected = homogeneous @ lidar2img.T
        depth = projected[:, 2]
        pixels = projected[:, :2] / np.maximum(depth[:, None], 1e-5)
        color = CLASS_COLORS[int(label) % len(CLASS_COLORS)]
        visible_points = []
        for first, second in BOX_EDGES:
            if depth[first] <= 0.5 or depth[second] <= 0.5:
                continue
            p1 = tuple(np.rint(pixels[first]).astype(int))
            p2 = tuple(np.rint(pixels[second]).astype(int))
            if ((-width <= p1[0] < 2 * width and
                 -height <= p1[1] < 2 * height) or
                    (-width <= p2[0] < 2 * width and
                     -height <= p2[1] < 2 * height)):
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
                visible_points.extend((p1, p2))
        if visible_points and score >= max(score_thr, 0.50):
            x = int(min(point[0] for point in visible_points))
            y = int(min(point[1] for point in visible_points)) - 5
            x = int(np.clip(x, 0, max(0, width - 1)))
            y = int(np.clip(y, 15, max(15, height - 1)))
            put_label(
                image, f'{CLASS_NAMES[int(label)]} {score:.2f}',
                (x, y), scale=0.40, color=color)


def bev_pixel(points: np.ndarray, size: int,
              point_cloud_range: Sequence[float]) -> np.ndarray:
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    u = (y_max - points[:, 1]) / (y_max - y_min) * (size - 1)
    v = (x_max - points[:, 0]) / (x_max - x_min) * (size - 1)
    return np.rint(np.stack((u, v), axis=1)).astype(np.int32)


def footprint(box: np.ndarray) -> np.ndarray:
    x, y, _, length, width, _, yaw = box[:7]
    local = np.asarray((
        (length / 2, width / 2),
        (length / 2, -width / 2),
        (-length / 2, -width / 2),
        (-length / 2, width / 2)), dtype=np.float32)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    return local @ rotation.T + np.asarray((x, y))


def render_bev(
        points: np.ndarray, gt_boxes: np.ndarray, prediction,
        point_cloud_range: Sequence[float], score_thr: float,
        size: int = 360) -> np.ndarray:
    canvas = np.full((size, size, 3), 247, dtype=np.uint8)
    in_range = (
        (points[:, 0] >= point_cloud_range[0]) &
        (points[:, 0] <= point_cloud_range[3]) &
        (points[:, 1] >= point_cloud_range[1]) &
        (points[:, 1] <= point_cloud_range[4]))
    selected = points[in_range]
    if len(selected) > 45000:
        selected = selected[::max(1, len(selected) // 45000)]
    pixels = bev_pixel(selected[:, :2], size, point_cloud_range)
    canvas[pixels[:, 1], pixels[:, 0]] = (105, 105, 105)

    for box in gt_boxes:
        polygon = bev_pixel(footprint(box), size, point_cloud_range)
        cv2.polylines(canvas, [polygon], True, (52, 168, 83), 2, cv2.LINE_AA)

    boxes = prediction.bboxes_3d.tensor.detach().cpu().numpy()
    scores = prediction.scores_3d.detach().cpu().numpy()
    labels = prediction.labels_3d.detach().cpu().numpy()
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thr:
            continue
        polygon = bev_pixel(footprint(box), size, point_cloud_range)
        cv2.polylines(
            canvas, [polygon], True,
            CLASS_COLORS[int(label) % len(CLASS_COLORS)], 2, cv2.LINE_AA)
    ego = bev_pixel(np.asarray(((0.0, 0.0),)), size, point_cloud_range)[0]
    cv2.circle(canvas, tuple(ego), 5, (20, 20, 20), -1, cv2.LINE_AA)
    put_label(canvas, 'BEV: predictions + ground truth', (10, 25), 0.53)
    put_label(canvas, 'green = ground truth', (10, 48), 0.42, (52, 168, 83))
    return canvas


def attention_rollout(fusion: Dict[str, torch.Tensor], box: np.ndarray,
                      point_cloud_range: Sequence[float],
                      grid_size: Tuple[int, int]):
    grid_h, grid_w = grid_size
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    grid_x = int(np.clip(
        np.floor((box[0] - x_min) / (x_max - x_min) * grid_w),
        0, grid_w - 1))
    grid_y = int(np.clip(
        np.floor((box[1] - y_min) / (y_max - y_min) * grid_h),
        0, grid_h - 1))
    query_index = grid_y * grid_w + grid_x
    attention = fusion['attention']
    camera_to_bottleneck = attention['image_bottleneck'][0, -1].mean(dim=0)
    bottleneck_to_lidar = attention['lidar_readout'][0].mean(dim=0)
    rollout = torch.matmul(
        bottleneck_to_lidar[query_index], camera_to_bottleneck)
    valid = fusion['image_valid_mask'][0, :, 0]
    rollout = rollout * valid.to(rollout.dtype)
    rollout = rollout / rollout.sum().clamp_min(1e-12)
    values = rollout.detach().cpu().numpy().reshape(grid_h, grid_w)
    positive = values[values > 0]
    entropy = float(-(positive * np.log(positive + 1e-12)).sum())
    entropy /= max(np.log(max(2, len(positive))), 1e-12)
    top_count = max(1, int(np.ceil(0.10 * len(positive))))
    top_mass = float(np.sort(positive)[-top_count:].sum())
    return values, dict(
        query_grid_x=grid_x,
        query_grid_y=grid_y,
        normalized_entropy=entropy,
        top_10_percent_mass=top_mass)


def render_attention(values: np.ndarray, query_box: np.ndarray,
                     point_cloud_range: Sequence[float],
                     statistics: dict, size: int = 360) -> np.ndarray:
    heat = np.flip(values.T, axis=(0, 1))
    positive = heat[heat > 0]
    high = np.percentile(positive, 99) if positive.size else 1.0
    normalized = np.clip(heat / max(float(high), 1e-12), 0, 1)
    normalized = cv2.resize(
        normalized, (size, size), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap(
        np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    canvas = np.uint8(0.88 * color + 0.12 * np.full_like(color, 245))
    polygon = bev_pixel(footprint(query_box), size, point_cloud_range)
    cv2.polylines(canvas, [polygon], True, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.polylines(canvas, [polygon], True, (20, 20, 20), 2, cv2.LINE_AA)
    put_label(canvas, 'MBT camera-to-LiDAR attention', (10, 25), 0.52)
    put_label(
        canvas,
        f'entropy {statistics["normalized_entropy"]:.3f}  '
        f'top-10% mass {statistics["top_10_percent_mass"]:.3f}',
        (10, 49), 0.39)
    return canvas


def run_model(model, dataset, index: int):
    item = dataset[index]
    display_images = [
        image.permute(1, 2, 0).cpu().numpy().clip(0, 255)
        .astype(np.uint8).copy()
        for image in item['inputs']['img']
    ]
    points = item['inputs']['points'].cpu().numpy().copy()
    eval_ann = item['data_samples'].eval_ann_info
    gt_boxes = eval_ann['gt_bboxes_3d'].tensor.cpu().numpy().copy()
    packed = pseudo_collate([item])
    batch = model.data_preprocessor(packed, training=False)
    inputs = batch['inputs']
    samples = batch['data_samples']
    with torch.no_grad():
        image_features = model.extract_img_feat(inputs['imgs'])
        lidar_features = list(model.extract_pts_feat(inputs['voxels'], 1))
        fusion = model.fusion_module(
            image_features,
            lidar_features[model.fusion_level],
            return_attention=True,
            **model._projection_metadata(samples, inputs['imgs']))
        lidar_features[model.fusion_level] = fusion['bev_features']
        prediction = model.bbox_head.predict(tuple(lidar_features), samples)[0]
    return display_images, points, gt_boxes, fusion, prediction, samples[0]


def render_frame(model, dataset, data_list: Sequence[dict], index: int,
                 point_cloud_range: Sequence[float], score_thr: float):
    images, points, gt_boxes, fusion, prediction, sample = run_model(
        model, dataset, index)
    scores = prediction.scores_3d.detach().cpu().numpy()
    labels = prediction.labels_3d.detach().cpu().numpy()
    corners = prediction.bboxes_3d.corners.detach().cpu().numpy()
    boxes = prediction.bboxes_3d.tensor.detach().cpu().numpy()
    camera_names = list(data_list[index]['images'])
    camera_panels = []
    for view, (camera_name, image) in enumerate(zip(camera_names, images)):
        lidar2img = compose_lidar2img(
            np.asarray(sample.metainfo['cam2img'][view]),
            np.asarray(sample.metainfo['lidar2cam'][view]))
        draw_projected_boxes(
            image, corners, scores, labels, lidar2img, score_thr)
        image = resize_cover(image, 640, 360)
        put_label(image, camera_name, (14, 29), 0.66)
        camera_panels.append(image)
    camera_mosaic = np.concatenate((
        np.concatenate(camera_panels[:3], axis=1),
        np.concatenate(camera_panels[3:], axis=1)), axis=0)

    bev = render_bev(
        points, gt_boxes, prediction, point_cloud_range, score_thr)
    valid_indices = np.flatnonzero(scores >= score_thr)
    if len(valid_indices):
        query_index = int(valid_indices[np.argmax(scores[valid_indices])])
    else:
        query_index = int(np.argmax(scores))
    attention, statistics = attention_rollout(
        fusion, boxes[query_index], point_cloud_range,
        model.fusion_module.token_grid_size)
    attention_panel = render_attention(
        attention, boxes[query_index], point_cloud_range, statistics)

    text_panel = np.full((360, 1200, 3), 247, dtype=np.uint8)
    token = data_list[index]['token']
    selected_label = CLASS_NAMES[int(labels[query_index])]
    lines = (
        'Corrected six-camera MBT fusion',
        f'validation index {index}  |  token {token[:16]}...',
        f'attention query: {selected_label}  score {scores[query_index]:.3f}',
        f'detections >= {score_thr:.2f}: {len(valid_indices)}',
        'Camera panels: predicted 3D boxes projected with nuScenes calibration',
        'BEV panel: LiDAR points, colored predictions, green ground truth',
        'Attention panel: last-layer LiDAR-query rollout through 4 bottlenecks',
    )
    for line_index, line in enumerate(lines):
        scale = 0.80 if line_index == 0 else 0.55
        weight = 2 if line_index == 0 else 1
        cv2.putText(
            text_panel, line, (36, 48 + line_index * 42),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), weight,
            cv2.LINE_AA)
    bottom = np.concatenate((bev, attention_panel, text_panel), axis=1)
    frame = np.concatenate((camera_mosaic, bottom), axis=0)
    statistics.update(dict(
        dataset_index=int(index),
        token=token,
        attention_query_class=selected_label,
        attention_query_score=float(scores[query_index]),
        detections_above_threshold=int(len(valid_indices))))
    return frame, statistics


def sample_score(detections: Iterable[dict], score_thr: float) -> float:
    selected = [
        detection for detection in detections
        if float(detection['detection_score']) >= score_thr
    ]
    classes = {detection['detection_name'] for detection in selected}
    confidence = sum(
        float(detection['detection_score']) for detection in selected[:20])
    return len(selected) + 1.5 * len(classes) + 0.1 * confidence


def select_static_indices(data_list: Sequence[dict], results: dict,
                          count: int, score_thr: float):
    scores = [
        sample_score(results.get(info['token'], ()), score_thr)
        for info in data_list
    ]
    boundaries = np.linspace(0, len(data_list), count + 1, dtype=int)
    indices = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        indices.append(int(start) + int(np.argmax(scores[start:end])))
    return indices, scores


def select_video_indices(data_list: Sequence[dict], scores: Sequence[float],
                         frame_count: int, nuscenes_root: str):
    from nuscenes.nuscenes import NuScenes

    nuscenes = NuScenes(
        version='v1.0-trainval', dataroot=nuscenes_root, verbose=False)
    index_by_token = {
        info['token']: index for index, info in enumerate(data_list)
    }
    best = None
    for start_token in index_by_token:
        sequence = []
        token = start_token
        while token and len(sequence) < frame_count:
            index = index_by_token.get(token)
            if index is None:
                break
            sequence.append(index)
            token = nuscenes.get('sample', token)['next']
        if len(sequence) != frame_count:
            continue
        window_score = float(np.mean([scores[index] for index in sequence]))
        if best is None or window_score > best[0]:
            best = (window_score, sequence)
    if best is None:
        raise RuntimeError('Could not find a contiguous nuScenes sequence')
    return best[1]


def contact_sheet(paths: Sequence[Path], output: Path) -> None:
    thumbnails = [
        cv2.resize(cv2.imread(str(path)), (960, 540),
                   interpolation=cv2.INTER_AREA)
        for path in paths
    ]
    if len(thumbnails) % 2:
        thumbnails.append(np.full_like(thumbnails[0], 247))
    rows = [
        np.concatenate(thumbnails[index:index + 2], axis=1)
        for index in range(0, len(thumbnails), 2)
    ]
    cv2.imwrite(str(output), np.concatenate(rows, axis=0),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    static_dir = output / 'static'
    static_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    data_list = mmengine.load(args.info_file)['data_list']
    persisted = json.loads(Path(args.results_json).read_text())['results']
    point_cloud_range = tuple(
        float(value) for value in model.fusion_module.point_cloud_range.cpu())

    static_indices, scores = select_static_indices(
        data_list, persisted, args.static_count, args.score_thr)
    summaries = []
    static_paths = []
    for index in static_indices:
        frame, summary = render_frame(
            model, dataset, data_list, index,
            point_cloud_range, args.score_thr)
        destination = static_dir / f'sample_{index:04d}.png'
        cv2.imwrite(str(destination), frame)
        static_paths.append(destination)
        summaries.append(summary)
        print(f'wrote {destination}', flush=True)
    contact_sheet(static_paths, output / 'qualitative_contact_sheet.jpg')

    video_indices = select_video_indices(
        data_list, scores, args.video_frames, args.nuscenes_root)
    video_path = output / 'nuscenes_fusion_sequence.mp4'
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*'mp4v'),
        args.video_fps, (1920, 1080))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video writer for {video_path}')
    for index in video_indices:
        frame, _ = render_frame(
            model, dataset, data_list, index,
            point_cloud_range, args.score_thr)
        writer.write(frame)
        print(f'video frame {index}', flush=True)
    writer.release()

    summary = dict(
        checkpoint=args.checkpoint,
        score_threshold=args.score_thr,
        static_samples=summaries,
        video_dataset_indices=video_indices,
        video_fps=args.video_fps)
    (output / 'visualization_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(f'wrote {video_path}')


if __name__ == '__main__':
    main()
