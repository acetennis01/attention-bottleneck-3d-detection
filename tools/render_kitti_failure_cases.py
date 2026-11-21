"""Illustrate KITTI errors from persisted predictions; never run training.

Matching follows the installed KITTI evaluator at one fixed score threshold,
not AP integration. Every frame's counts are checked against its evaluator.
Candidate selection is intentionally post-hoc and is not a frequency estimate.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mmengine
import numpy as np
from mmdet3d.evaluation.functional.kitti_utils.eval import (
    clean_data, compute_statistics_jit, d3_box_overlap, image_box_overlap)
from mmdet3d.structures import CameraInstance3DBoxes


NAMES = ('Car', 'Pedestrian', 'Cyclist')
COLORS = {'miss': (210, 30, 190), 'false_positive': (30, 35, 230),
          'localization': (20, 135, 240)}
TITLES = {'miss': 'Missed object', 'false_positive': 'Unmatched prediction',
          'localization': '3D localization error'}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def annotation(info, names):
    inst = info['instances']
    boxes = np.asarray([x['bbox_3d'] for x in inst], np.float64).reshape(-1, 7)
    return dict(
        name=np.array([names[int(x['bbox_label_3d'])] for x in inst]),
        bbox=np.asarray([x['bbox'] for x in inst], np.float64).reshape(-1, 4),
        location=boxes[:, :3], dimensions=boxes[:, 3:6], rotation_y=boxes[:, 6],
        truncated=np.array([x['truncated'] for x in inst]),
        occluded=np.array([x['occluded'] for x in inst]),
        alpha=np.array([x['alpha'] for x in inst]))


def boxes_array(anno):
    return np.concatenate((anno['location'], anno['dimensions'],
                           anno['rotation_y'][:, None]), axis=1).astype(np.float64)


def trace_matches(overlap, gt_flags, dt_flags, scores, cutoff, minimum):
    """KITTI compute_fp=True greedy assignment, returning object identities."""
    used, true_pairs, missed = set(), [], []
    for i, flag in enumerate(gt_flags):
        if flag == -1:
            continue
        chosen, best, chosen_ignored = -1, 0., False
        for j, dt_flag in enumerate(dt_flags):
            if dt_flag == -1 or j in used or scores[j] < cutoff:
                continue
            value = overlap[j, i]
            if value > minimum and (value > best or chosen_ignored) and dt_flag == 0:
                chosen, best, chosen_ignored = j, value, False
            elif value > minimum and chosen == -1 and dt_flag == 1:
                chosen, chosen_ignored = j, True
        if chosen == -1:
            if flag == 0:
                missed.append(i)
        else:
            used.add(chosen)
            if flag == 0 and dt_flags[chosen] == 0:
                true_pairs.append((i, chosen))
    false = [j for j, flag in enumerate(dt_flags)
             if flag == 0 and j not in used and scores[j] >= cutoff]
    return true_pairs, false, missed


def scan(infos, predictions, names, cutoff):
    totals = dict(tp=0, fp=0, fn=0)
    candidates = []
    for index, (info, dt) in enumerate(zip(infos, predictions)):
        sid = int(info['sample_idx'])
        # Persisted MMEngine sample_idx is the positional index, not frame ID.
        if len(dt['sample_idx']) and not np.all(np.asarray(dt['sample_idx']) == index):
            raise ValueError(f'Unexpected prediction ordering at {index}')
        gt = annotation(info, names)
        gb, db = boxes_array(gt), boxes_array(dt)
        valid = np.flatnonzero((gb[:, 3:6] > 0).all(1))
        overlap = np.zeros((len(db), len(gb)), np.float64)
        if len(db) and len(valid):
            overlap[:, valid] = d3_box_overlap(db, gb[valid])
        overlap2d = image_box_overlap(dt['bbox'].astype(np.float64), gt['bbox'])
        for class_id, name in enumerate(NAMES):
            _, ig, idt, dc = clean_data(gt, dt, class_id, 1)
            ig, idt = np.array(ig), np.array(idt)
            minimum = .7 if name == 'Car' else .5
            pairs, false, missed = trace_matches(
                overlap, ig, idt, dt['score'], cutoff, minimum)
            gd = np.column_stack((gt['bbox'], gt['alpha']))
            dd = np.column_stack((dt['bbox'], dt['alpha'], dt['score']))
            official = compute_statistics_jit(
                overlap, gd, dd, ig, idt,
                np.asarray(dc, np.float64).reshape(-1, 4),
                2, minimum, cutoff, True, False)
            if tuple(map(len, (pairs, false, missed))) != tuple(official[:3]):
                raise ValueError(f'Matcher disagrees with KITTI: {sid}, {name}')
            for key, value in zip(('tp', 'fp', 'fn'), official[:3]):
                totals[key] += int(value)
            same = (dt['name'] == name) & (dt['score'] >= cutoff)
            for g in missed:
                box = gt['bbox'][g]
                size = box[2:] - box[:2]
                distance = float(np.linalg.norm(gb[g, [0, 2]]))
                best3 = float(overlap[same, g].max(initial=0))
                best2 = float(overlap2d[same, g].max(initial=0))
                # Readable, complete misses, not merely inaccurate 3D boxes.
                if (size[1] >= 45 and size[0] >= 18 and 4 < distance < 40
                        and best3 < .1 and best2 < .1 and gt['truncated'][g] <= .1):
                    candidates.append(dict(
                        kind='miss', sample_idx=sid, dataset_index=index,
                        class_name=name, gt_index=int(g), pred_index=None,
                        maximum_same_class_3d_iou=best3,
                        maximum_same_class_2d_iou=best2, distance_m=distance,
                        rank=float(min(size[1], 180) * min(size[0], 100))))
            for d in false:
                box = dt['bbox'][d]
                size = box[2:] - box[:2]
                if min(size) < 22 or size[1] < 45:
                    continue
                best = int(overlap2d[d].argmax()) if len(gb) else None
                any2 = float(overlap2d[d].max(initial=0))
                any3 = float(overlap[d].max(initial=0))
                score = float(dt['score'][d])
                # Extra display-only safeguards: exclude overlap with ANY
                # annotated class or DontCare for apparent-background examples.
                coverage = image_box_overlap(
                    dt['bbox'][d:d+1].astype(np.float64), gt['bbox'], 0)
                if any2 < .1 and any3 < .1 and coverage.max(initial=0) < .3:
                    kind, g = 'false_positive', None
                elif (best is not None and gt['name'][best] == name
                      and ig[best] == 0 and overlap2d[d, best] >= .5
                      and .05 < overlap[d, best] < minimum):
                    kind, g = 'localization', best
                else:
                    continue
                candidates.append(dict(
                    kind=kind, sample_idx=sid, dataset_index=index,
                    class_name=name, gt_index=g, pred_index=int(d),
                    confidence=score, maximum_any_gt_2d_iou=any2,
                    maximum_any_gt_3d_iou=any3,
                    matched_3d_iou=float(overlap[d, g]) if g is not None else None,
                    rank=float(score * min(size[1], 160) * min(size[0], 150))))
        if (index+1) % 500 == 0:
            print(f'Checked {index+1}/{len(infos)} frames', flush=True)
    return candidates, totals


def text(image, message, xy, scale=.65, color=(30, 30, 30), thickness=1):
    cv2.putText(image, message, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def camera_corners(array):
    return CameraInstance3DBoxes(np.asarray(array, np.float32).reshape(-1, 7)).corners.numpy()


def draw_camera(image, corners, matrix, color):
    for box in corners:
        hom = np.column_stack((box, np.ones(8))) @ matrix.T
        uv = hom[:, :2] / np.maximum(hom[:, 2:3], 1e-6)
        for a, b in EDGES:
            if min(hom[[a, b], 2]) <= 1e-5:
                continue
            p, q = np.clip(uv[[a, b]], -1e5, 1e5).round().astype(int)
            yes, p, q = cv2.clipLine((0, 0, image.shape[1], image.shape[0]),
                                    tuple(p), tuple(q))
            if yes:
                cv2.line(image, p, q, color, 3, cv2.LINE_AA)


def render(record, info, dt, names, root, output):
    sid, kind = record['sample_idx'], record['kind']
    original = cv2.imread(str(root/'training/image_2'/f'{sid:06d}.png'))
    if original is None:
        raise FileNotFoundError(sid)
    image = original.copy()
    gt = annotation(info, names)
    gb, db = boxes_array(gt), boxes_array(dt)
    color = COLORS[kind]
    g, d = record['gt_index'], record['pred_index']
    matrix = np.asarray(info['images']['CAM2']['cam2img'])
    if g is not None:
        draw_camera(image, camera_corners(gb[g:g+1]), matrix, (210, 30, 190))
        rect = gt['bbox'][g].round().astype(int)
    else:
        rect = dt['bbox'][d].round().astype(int)
    if d is not None:
        draw_camera(image, camera_corners(db[d:d+1]), matrix, color)
    # Rectangle aids visibility; the 3D boxes above use calibrated corners.
    cv2.rectangle(image, tuple(rect[:2]), tuple(rect[2:]), color, 2)
    label = record['class_name'] + (' GT' if kind == 'miss' else f' {record["confidence"]:.2f}')
    x, y = int(max(4, rect[0])), int(max(24, rect[1]-7))
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4, cv2.LINE_AA)
    text(image, label, (x, y), .65, color, 2)
    # Full camera view above a plainly identified magnified crop.
    width = 1400
    camera = cv2.resize(image, (width, round(width*image.shape[0]/image.shape[1])))
    center = (rect[:2] + rect[2:])/2
    crop_w = max(180, float(rect[2]-rect[0])*2.4)
    crop_h = max(140, float(rect[3]-rect[1])*1.7)
    x0 = max(0, int(center[0]-crop_w/2)); x1 = min(image.shape[1], int(center[0]+crop_w/2))
    y0 = max(0, int(center[1]-crop_h/2)); y1 = min(image.shape[0], int(center[1]+crop_h/2))
    detail = image[y0:y1, x0:x1]
    lower = np.full((310, width, 3), 250, np.uint8)
    ratio = min(520/detail.shape[1], 265/detail.shape[0])
    detail = cv2.resize(detail, None, fx=ratio, fy=ratio)
    lower[35:35+detail.shape[0], 15:15+detail.shape[1]] = detail
    text(lower, 'Magnified detail (same image)', (15, 25), .55)
    text(lower, 'ABFuse | ResNet-50/FPN | 8 bottleneck tokens | epoch 14', (560, 45), .6, thickness=2)
    text(lower, 'Prediction confidence threshold: 0.30', (560, 85), .6)
    text(lower, 'KITTI moderate, class-specific 3D IoU matching', (560, 120), .6)
    if kind == 'miss':
        lines = ['Magenta: annotated object with no matching prediction.',
                 f'Best same-class 3D IoU: {record["maximum_same_class_3d_iou"]:.3f}',
                 'No overlapping same-class detection at score >= 0.30.']
    elif kind == 'false_positive':
        lines = ['Red: prediction without a matching annotation.',
                 f'Best 3D IoU with any GT: {record["maximum_any_gt_3d_iou"]:.3f}',
                 'Outside overlapping annotated objects / DontCare regions.']
    else:
        lines = ['Orange: prediction. Magenta: annotated 3D box.',
                 f'3D IoU: {record["matched_3d_iou"]:.3f} (below class threshold)',
                 'Object found in the image, but its 3D box is inaccurate.']
    for line, y in zip(lines, (170, 205, 240)):
        text(lower, line, (560, y), .55)
    banner = np.full((65, width, 3), 255, np.uint8)
    text(banner, f'KITTI {sid:06d} | {TITLES[kind]}: {record["class_name"]}', (18, 40), .85, thickness=2)
    result = np.concatenate((banner, camera, lower))
    filename = f'{kind}_{sid:06d}_{record["class_name"].lower()}.png'
    if not cv2.imwrite(str(output/filename), result):
        raise RuntimeError(filename)
    return result, dict(**record, filename=filename, sha256=sha256(output/filename),
                        crop_xyxy=[x0, y0, x1, y1])


def assemble_existing(output, filenames):
    """Stack unchanged example panels, recording explicit manual selection."""
    manifest_path = output/'failure_cases_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    known = {x['filename']: x for x in manifest['examples']}
    rows = []
    for filename in filenames:
        if filename not in known or sha256(output/filename) != known[filename]['sha256']:
            raise ValueError(f'Unknown or changed source image: {filename}')
        rows.append(cv2.imread(str(output/filename)))
    width = rows[0].shape[1]
    parts = []
    for i, row in enumerate(rows):
        if i:
            parts.append(np.full((18, width, 3), 255, np.uint8))
        parts.append(row)
    destination = output/'ABFuse_failure_examples.png'
    if destination.exists():
        raise FileExistsError(destination)
    cv2.imwrite(str(destination), np.concatenate(parts))
    manifest['selected_figure'] = dict(
        filename=destination.name, sha256=sha256(destination), sources=filenames,
        selection='Manual post-hoc selection for visible pedestrian miss, trash-bin false car prediction, and car localization failure. Panels are unchanged.')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path)
    p.add_argument('--data-root', default='data/kitti', type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--per-kind', default=6, type=int)
    p.add_argument('--assemble-existing', nargs='+', metavar='FILENAME')
    args = p.parse_args()
    if args.assemble_existing:
        assemble_existing(args.output_dir, args.assemble_existing)
        return
    if args.report is None:
        p.error('--report is required to scan predictions')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text())
    if report['status'] != 'passed' or report['mode'] != 'correct' or report['validation_samples'] != 3769:
        raise ValueError('Expected a passed, full KITTI correct-camera validation')
    if report['checkpoint_sha256'] != '308921f9d01c2eec8ee6633983fbac8c688b64489a8840858379bb3f1a8263b0':
        raise ValueError('Expected the larger eight-token epoch-14 model')
    info_path = args.data_root/'kitti_infos_val.pkl'
    infos = mmengine.load(info_path)
    predictions = mmengine.load(report['predictions'])
    data = infos['data_list']
    if len(data) != len(predictions) or len(data) != 3769:
        raise ValueError('Incomplete validation records')
    names = {v: k for k, v in infos['metainfo']['categories'].items()}
    candidates, totals = scan(data, predictions, names, .30)
    examples, seen = [], set()
    for kind in COLORS:
        pool = sorted([r for r in candidates if r['kind'] == kind],
                      key=lambda r: -r['rank'])
        rows = []
        for record in pool:
            if record['sample_idx'] in seen:
                continue
            seen.add(record['sample_idx'])
            i = record['dataset_index']
            row, item = render(record, data[i], predictions[i], names,
                               args.data_root, args.output_dir)
            rows.append(row); examples.append(item)
            if len(rows) >= args.per_kind:
                break
        if rows:
            thumbs = [cv2.resize(row, (1050, round(row.shape[0]*.75))) for row in rows]
            cv2.imwrite(str(args.output_dir/f'{kind}_contact_sheet.png'), np.concatenate(thumbs))
    manifest = dict(
        model='ABFuse larger KITTI, eight tokens, epoch 14',
        renderer_sha256=sha256(__file__),
        source_report=str(args.report), checkpoint=report['checkpoint'],
        checkpoint_sha256=report['checkpoint_sha256'],
        predictions=report['predictions'], predictions_sha256=sha256(report['predictions']),
        info_sha256=sha256(info_path), validation_samples=len(data),
        score_threshold=.30, difficulty='moderate', metric='3D IoU',
        matching_thresholds={'Car': .7, 'Pedestrian': .5, 'Cyclist': .5},
        fixed_threshold_totals=totals, matching_verified_against_installed_evaluator=True,
        selection='Post-hoc ranked readable failure examples, not representative sampling or AP estimates. Background-style candidates additionally exclude overlap with any annotated class and DontCare. Localization failures are separately identified.',
        caveat='A KITTI 3D false positive means unmatched under its scoring rules, not necessarily a nonexistent physical object. Ignored GT are respected; DontCare suppression is used by the official evaluator only for 2D, with extra exclusion here for display candidates.',
        candidate_count=len(candidates), candidates=candidates, examples=examples)
    (args.output_dir/'failure_cases_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Rendered {len(examples)} examples; verified {len(data)} frames: {totals}', flush=True)


if __name__ == '__main__':
    main()
