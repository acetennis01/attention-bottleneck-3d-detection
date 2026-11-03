"""Regenerate paper figures from strictly loaded KITTI checkpoints.

Run from the MMDetection3D root. Original images/experiments are never edited.
The attention figure is a last-layer, head-averaged attention-path product,
not a gradient attribution or a complete multi-layer attention rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mmengine
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from mmdet3d.structures import Box3DMode, CameraInstance3DBoxes

from projects.myfusion.tools.render_attention_maps import (
    CLASS_COLORS, CLASS_NAMES, apply_heatmap, attention_rollout,
    camera_heatmap, resize_and_pad, robust_normalize,
    run_model_with_attention, titled)


FOOTPRINT = [0, 3, 7, 4]  # LiDARInstance3DBoxes bottom face, not [:4].
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_verified(report_path, device):
    report = json.loads(Path(report_path).read_text())
    if report.get('status') != 'passed':
        raise ValueError('A passed full-validation report is required')
    checkpoint = Path(report['checkpoint'])
    actual_sha = sha256(checkpoint)
    if actual_sha != report['checkpoint_sha256']:
        raise ValueError(f'Checkpoint digest mismatch: {checkpoint}')
    cfg = Config.fromfile(report['source_config'])
    model = init_model(cfg, checkpoint=None, device=device)
    saved = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    model.eval()
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    if len(dataset) != 3769:
        raise ValueError('Expected the full, unchanged KITTI validation split')
    return model, dataset, cfg, dict(
        checkpoint=str(checkpoint), checkpoint_sha256=actual_sha,
        source_config=report['source_config'], strict_checkpoint_load=True,
        validation_samples=len(dataset))


def text(image, label, xy, scale=.55, color=(30, 30, 30), thickness=1):
    cv2.putText(image, label, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def pixels(xy, extent, size):
    x0, y0, x1, y1 = extent
    return np.rint(np.column_stack((
        (y1 - xy[:, 1]) / (y1 - y0) * (size - 1),
        (x1 - xy[:, 0]) / (x1 - x0) * (size - 1)))).astype(np.int32)


def draw_points(panel, points, extent, color):
    x0, y0, x1, y1 = extent
    keep = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
            (points[:, 1] >= y0) & (points[:, 1] <= y1))
    p = pixels(points[keep, :2], extent, panel.shape[0])
    panel[p[:, 1], p[:, 0]] = color


def draw_boxes(panel, boxes, labels, extent, selected=None):
    corners = boxes.corners.detach().cpu().numpy()
    for index, (box, label) in enumerate(zip(corners, labels)):
        p = pixels(box[FOOTPRINT, :2], extent, panel.shape[0])
        color = CLASS_COLORS[int(label)]
        cv2.polylines(panel, [p], True, color, 2, cv2.LINE_AA)
        # Short heading marker: center -> positive local x face.
        center = p.mean(0).round().astype(int)
        front = p[[2, 3]].mean(0).round().astype(int)
        cv2.line(panel, tuple(center), tuple(front), color, 2, cv2.LINE_AA)
        if selected == index:
            cv2.polylines(panel, [p], True, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.polylines(panel, [p], True, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.circle(panel, tuple(center), 6, (20, 20, 20), -1)
            cv2.circle(panel, tuple(center), 3, (255, 255, 255), -1)


def ground_truth(info):
    instances = [x for x in info['instances']
                 if int(x['bbox_label_3d']) in (0, 1, 2)]
    boxes = CameraInstance3DBoxes(
        np.asarray([x['bbox_3d'] for x in instances], np.float32).reshape(-1, 7))
    boxes = boxes.convert_to(
        Box3DMode.LIDAR,
        np.linalg.inv(np.asarray(info['images']['CAM2']['lidar2cam'])))
    labels = np.asarray([x['bbox_label_3d'] for x in instances], int)
    return boxes, labels, instances


def selected_prediction(prediction, threshold, extent):
    boxes = prediction.bboxes_3d
    centers = boxes.tensor.detach().cpu().numpy()
    scores = prediction.scores_3d.detach().cpu().numpy()
    keep = ((scores >= threshold) & (centers[:, 0] >= extent[0]) &
            (centers[:, 0] <= extent[2]) & (centers[:, 1] >= extent[1]) &
            (centers[:, 1] <= extent[3]))
    return boxes[keep], prediction.labels_3d.detach().cpu().numpy()[keep], scores[keep]


def header(panel, title, subtitle=''):
    banner = np.full((56, panel.shape[1], 3), 250, np.uint8)
    text(banner, title, (9, 22), .53, thickness=2)
    if subtitle:
        text(banner, subtitle, (9, 44), .38)
    return np.concatenate((banner, panel))


def bev_panel(points, boxes, labels, extent, title, size=360):
    panel = np.full((size, size, 3), 248, np.uint8)
    for x in np.arange(0, 71, 10):
        v = pixels(np.array([[x, extent[1]], [x, extent[3]]]), extent, size)
        cv2.line(panel, tuple(v[0]), tuple(v[1]), (225, 225, 225), 1)
    for y in np.arange(-30, 31, 10):
        v = pixels(np.array([[extent[0], y], [extent[2], y]]), extent, size)
        cv2.line(panel, tuple(v[0]), tuple(v[1]), (225, 225, 225), 1)
    draw_points(panel, points, extent, (170, 170, 170))
    draw_boxes(panel, boxes, labels, extent)
    origin = pixels(np.array([[0., 0.]]), extent, size)[0]
    cv2.circle(panel, tuple(origin), 4, (55, 55, 55), -1)
    return header(panel, f'{title} ({len(boxes)})', 'Front up; grid spacing 10 m')


def camera_boxes(image, boxes, labels, scores, matrix):
    panel = image.copy()
    for box, label, score in zip(boxes.corners.detach().cpu().numpy(), labels, scores):
        hom = np.column_stack((box, np.ones(8))) @ matrix.T
        depth = hom[:, 2]
        uv = hom[:, :2] / np.maximum(depth[:, None], 1e-6)
        # Clipping individual segments prevents huge off-screen coordinates.
        bounds = (0, 0, panel.shape[1], panel.shape[0])
        color = CLASS_COLORS[int(label)]
        for a, b in EDGES:
            if min(depth[a], depth[b]) <= 1e-5:
                continue
            p, q = np.clip(uv[[a, b]], -1e5, 1e5).round().astype(int)
            valid, p, q = cv2.clipLine(bounds, tuple(p), tuple(q))
            if valid:
                cv2.line(panel, p, q, color, 2, cv2.LINE_AA)
        if np.all(depth > 1e-5):
            xy = uv.min(0).round().astype(int)
            x = int(np.clip(xy[0], 2, panel.shape[1] - 135))
            y = int(np.clip(xy[1] - 5, 18, panel.shape[0] - 2))
            text(panel, f'{CLASS_NAMES[int(label)]} {score:.2f}',
                 (x, y), .48, color, 2)
    return panel


def qualitative_row(info, lidar_prediction, fusion_prediction, args, extent):
    sample_id = int(info['sample_idx'])
    root = Path(args.data_root) / 'training'
    full = np.fromfile(root / 'velodyne' / f'{sample_id:06d}.bin', np.float32).reshape(-1, 4)
    reduced = np.fromfile(root / 'velodyne_reduced' / f'{sample_id:06d}.bin', np.float32).reshape(-1, 4)
    # Identical PointsRangeFilter bounds to the model pipeline.
    pc = args.point_cloud_range
    keep = ((reduced[:, :3] >= np.array(pc[:3])).all(1) &
            (reduced[:, :3] < np.array(pc[3:])).all(1))
    reduced = reduced[keep]
    image = cv2.imread(str(root / 'image_2' / f'{sample_id:06d}.png'))
    if image is None:
        raise FileNotFoundError(sample_id)
    reference = np.full((360, 360, 3), 15, np.uint8)
    draw_points(reference, full, (-40, -60, 80, 60), (110, 110, 110))
    draw_points(reference, reduced, (-40, -60, 80, 60), (25, 175, 255))
    origin = pixels(np.array([[0., 0.]]), (-40, -60, 80, 60), 360)[0]
    cv2.circle(reference, tuple(origin), 4, (70, 210, 70), -1)
    panels = [header(reference, 'LiDAR 360 deg reference', 'Orange: camera-FOV model input')]
    counts = {}
    for name, prediction in [('LiDAR-only', lidar_prediction), ('ABFuse', fusion_prediction)]:
        boxes, labels, scores = selected_prediction(prediction, args.score_thr, extent)
        panels.append(bev_panel(reduced, boxes, labels, extent, name))
        counts[name] = len(boxes)
        if name == 'ABFuse':
            camera = camera_boxes(image, boxes, labels, scores,
                                  np.asarray(info['images']['CAM2']['lidar2img']))
    gt, labels, _ = ground_truth(info)
    centers = gt.tensor.numpy()
    valid = ((centers[:, 0] >= extent[0]) & (centers[:, 0] <= extent[2]) &
             (centers[:, 1] >= extent[1]) & (centers[:, 1] <= extent[3]))
    panels.append(bev_panel(reduced, gt[valid], labels[valid], extent, 'Ground truth'))
    banner = np.full((52, 1440, 3), 255, np.uint8)
    text(banner, f'KITTI {sample_id:06d} | ABFuse: ResNet-50, 8 tokens | score >= {args.score_thr:.2f}',
         (12, 31), .66, thickness=2)
    for label, x in [(2, 1050), (0, 1150), (1, 1300)]:
        cv2.rectangle(banner, (x, 18), (x + 14, 32), CLASS_COLORS[label], -1)
        text(banner, CLASS_NAMES[label], (x + 20, 31), .48)
    camera = cv2.resize(camera, (1440, round(1440 * image.shape[0] / image.shape[1])))
    camera = header(camera, 'ABFuse predictions projected into the camera image')
    return np.concatenate((banner, np.concatenate(panels, 1), camera)), dict(
        sample_idx=sample_id, predicted_counts=counts, ground_truth_count=int(valid.sum()),
        raw_points=len(full), model_input_points=len(reduced))


def attention_example(info, label, fusion, prediction, args, extent):
    gt, gt_labels, instances = ground_truth(info)
    candidates = [i for i, x in enumerate(instances)
                  if gt_labels[i] == label and x['difficulty'] == 1]
    pred_ids = torch.nonzero(
        (prediction.labels_3d == label) &
        (prediction.scores_3d >= args.score_thr)).flatten()
    if not candidates or not len(pred_ids):
        raise ValueError(f'No moderate {CLASS_NAMES[label]} detection in {info["sample_idx"]}')
    iou = prediction.bboxes_3d.overlaps(
        prediction.bboxes_3d[pred_ids], gt[candidates].to(args.device))
    matched = torch.nonzero(iou >= (.7 if label == 2 else .5))
    if not len(matched):
        raise ValueError(f'No 3D-IoU-matched {CLASS_NAMES[label]} true positive in {info["sample_idx"]}')
    row, col = max(matched.tolist(), key=lambda ij: float(prediction.scores_3d[pred_ids[ij[0]]]))
    selected = int(pred_ids[row])
    instance = instances[candidates[col]]
    box = prediction.bboxes_3d.tensor[selected].detach().cpu().numpy()
    weights, query, stats = attention_rollout(
        fusion, box, args.point_cloud_range, args.grid_size)
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1., atol=1e-5):
        raise ValueError('Attention is not finite and normalized')
    height_weights = fusion['attention']['image_height_weights'][0].detach().cpu().numpy()
    sid = int(info['sample_idx'])
    image = cv2.imread(str(Path(args.data_root) / 'training/image_2' / f'{sid:06d}.png'))
    heat = camera_heatmap(weights, height_weights,
                          np.asarray(info['images']['CAM2']['lidar2img']),
                          image.shape[:2], args.point_cloud_range, args.heights)
    camera = apply_heatmap(image, heat)
    x1, y1, x2, y2 = np.rint(instance['bbox']).astype(int)
    cv2.rectangle(camera, (x1, y1), (x2, y2), (255, 255, 255), 4, cv2.LINE_AA)
    cv2.rectangle(camera, (x1, y1), (x2, y2), (20, 20, 20), 2, cv2.LINE_AA)
    camera = titled(resize_and_pad(camera, 1050, 525),
                    f'{CLASS_NAMES[label]} -- ABFuse camera attention',
                    f'KITTI {sid:06d} | 8 bottleneck tokens | white = selected ground truth')
    bev = np.full((525, 525, 3), 245, np.uint8)
    heat_bev = cv2.resize(robust_normalize(np.flip(weights.T, (0, 1))),
                          (525, 525), interpolation=cv2.INTER_LINEAR)
    bev = apply_heatmap(bev, np.clip(heat_bev, 0, 1), alpha=.66)
    points = np.fromfile(Path(args.data_root) / 'training/velodyne_reduced' /
                         f'{sid:06d}.bin', np.float32).reshape(-1, 4)
    draw_points(bev, points, extent, (65, 65, 65))
    boxes, labels, _ = selected_prediction(prediction, args.score_thr, extent)
    draw_boxes(bev, boxes, labels, extent)
    draw_boxes(bev, prediction.bboxes_3d[selected:selected+1], [label], extent, selected=0)
    score = float(prediction.scores_3d[selected])
    bev = titled(bev, 'BEV attention and predictions',
                 f'confidence={score:.3f}; white = matched detection')
    np.savez_compressed(args.output_dir / f'attention_{sid:06d}.npz',
                        spatial_attention=weights, height_weights=height_weights,
                        image_bottleneck=fusion['attention']['image_bottleneck'].cpu().numpy(),
                        lidar_readout=fusion['attention']['lidar_readout'].cpu().numpy())
    return np.concatenate((camera, bev), 1), dict(
        sample_idx=sid, class_name=CLASS_NAMES[label], prediction_score=score,
        selected_prediction_index=selected, matched_3d_iou=float(iou[row, col]),
        query_token_index=query, **stats)


def stack(rows):
    separator = np.full((18, rows[0].shape[1], 3), 255, np.uint8)
    return np.concatenate([part for i, row in enumerate(rows)
                           for part in ([separator, row] if i else [row])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fusion-report', required=True)
    parser.add_argument('--baseline-report', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', default='data/kitti')
    parser.add_argument('--qualitative-samples', nargs=3, type=int, default=[8, 343, 767])
    parser.add_argument('--attention-samples', nargs=3, type=int, default=[6804, 2206, 767])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=.30)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new output directory; existing figures are preserved')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.50)
    model, dataset, cfg, model_record = load_verified(args.fusion_report, args.device)
    baseline, baseline_dataset, _, baseline_record = load_verified(args.baseline_report, args.device)
    args.point_cloud_range = model.fusion_module.point_cloud_range.detach().cpu().tolist()
    args.grid_size = tuple(model.fusion_module.token_grid_size)
    args.heights = model.fusion_module.height_anchors.detach().cpu().tolist()
    if tuple(model.fusion_module.bottleneck_tokens.shape[1:]) != (8, 256):
        raise ValueError('This figure recipe describes the larger eight-token KITTI model')
    infos = mmengine.load(Path(args.data_root) / 'kitti_infos_val.pkl')['data_list']
    index = {int(x['sample_idx']): i for i, x in enumerate(infos)}
    extent = (args.point_cloud_range[0], args.point_cloud_range[1],
              args.point_cloud_range[3], args.point_cloud_range[4])
    predictions, fused, validations = {}, {}, []
    for sid in sorted(set(args.qualitative_samples + args.attention_samples)):
        i = index[sid]
        # MMEngine replaces sample_idx with the positional dataset index.
        # The KITTI frame identity is preserved in the resolved file paths.
        if int(Path(dataset.get_data_info(i)['lidar_path']).stem) != sid:
            raise ValueError('Dataset and annotation order differ')
        with torch.no_grad():
            fusion, prediction, _ = run_model_with_attention(model, dataset, i)
            ordinary = model.test_step(pseudo_collate([dataset[i]]))[0].pred_instances_3d
        if len(ordinary.scores_3d) != len(prediction.scores_3d):
            raise ValueError('Attention capture changed the detection count')
        scores_equal = torch.allclose(ordinary.scores_3d, prediction.scores_3d, atol=1e-4, rtol=1e-4)
        boxes_equal = torch.allclose(ordinary.bboxes_3d.tensor, prediction.bboxes_3d.tensor, atol=1e-3, rtol=1e-4)
        if not scores_equal or not boxes_equal:
            raise ValueError('Attention capture changed ordinary inference beyond tolerance')
        if model.fusion_module.training or not fusion['lidar_token_keep_mask'].all():
            raise ValueError('Training or LiDAR masking active during visualization')
        if not torch.isfinite(fusion['bev_features']).all():
            raise ValueError('Non-finite fusion output')
        predictions[sid], fused[sid] = prediction, fusion
        validations.append(dict(sample_idx=sid, attention_inference_matches_standard=True,
                                maximum_score_difference=float((ordinary.scores_3d-prediction.scores_3d).abs().max())))
        print(f'ABFuse inference verified: {sid:06d}', flush=True)
    qual_rows, qual_records = [], []
    for sid in args.qualitative_samples:
        i = index[sid]
        if int(Path(baseline_dataset.get_data_info(i)['lidar_path']).stem) != sid:
            raise ValueError('Baseline dataset order differs')
        with torch.no_grad():
            lidar = baseline.test_step(pseudo_collate([baseline_dataset[i]]))[0].pred_instances_3d
        row, record = qualitative_row(infos[i], lidar, predictions[sid], args, extent)
        qual_rows.append(row)
        qual_records.append(record)
    attention_rows, attention_records = [], []
    for label, sid in enumerate(args.attention_samples):
        row, record = attention_example(infos[index[sid]], label, fused[sid], predictions[sid], args, extent)
        attention_rows.append(row)
        attention_records.append(record)
    outputs = {'qualitative_midpoint_examples.png': stack(qual_rows),
               'attention_maps_camera_bev.png': stack(attention_rows)}
    for filename, image in outputs.items():
        if not cv2.imwrite(str(args.output_dir / filename), image):
            raise RuntimeError(filename)
    manifest = dict(
        model='ABFuse: ResNet-50/FPN, 256-D, eight bottleneck tokens, epoch 14',
        fusion=model_record, lidar_baseline=baseline_record,
        score_threshold=args.score_thr, qualitative_examples=qual_records,
        attention_examples=attention_records, inference_checks=validations,
        example_selection='Road scenes selected before inference; prior attention sample IDs retained. No selection by fusion gain.',
        attention_method='Head-averaged final-layer LiDAR-to-bottleneck attention multiplied by bottleneck-to-camera attention; valid cells renormalized. Camera heat uses learned height weights and calibration. Each heatmap independently normalized for display.',
        caveat='Illustrative attention paths, not a full multi-layer rollout or causal attribution. Raw 360-degree scans are reference-only; inference uses camera-FOV reduced points with the configured range filter.',
        outputs={name: dict(shape=list(image.shape), sha256=sha256(args.output_dir/name))
                 for name, image in outputs.items()})
    (args.output_dir / 'visualization_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
