"""Render alternative, explicitly selected KITTI attention examples.

Selection is for illustration, not a representative or quantitative study.
The checkpoint, attention extraction, normalization, and smoothing are unchanged.
"""

import argparse
import json
from pathlib import Path

import cv2
import mmengine
import numpy as np
import torch
from mmengine.dataset import pseudo_collate

from projects.myfusion.tools.render_attention_maps import run_model_with_attention
from projects.myfusion.tools.render_kitti_paper_figures import (
    CLASS_NAMES, attention_example, load_verified, sha256, stack)


def assemble_recommended(output_dir):
    """Assemble unchanged rendered rows selected for visual clarity.

    This is explicitly post-hoc illustration selection, not evaluation.
    All candidate rows and original triplet options remain available.
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir/'attention_alternatives_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    samples = [6475, 4092, 5010]
    records = {x['sample_idx']: x for x in manifest['examples']}
    rows = [cv2.imread(str(output_dir/records[sid]['filename']))
            for sid in samples]
    if any(row is None for row in rows):
        raise FileNotFoundError('A recommended source row is missing')
    name = 'attention_maps_camera_bev_recommended.png'
    if (output_dir/name).exists():
        raise FileExistsError(output_dir/name)
    if not cv2.imwrite(str(output_dir/name), stack(rows)):
        raise RuntimeError(name)
    manifest['outputs'][name] = dict(
        samples=samples, sha256=sha256(output_dir/name),
        selection='Post-hoc manual selection for clear street scenes, visible matched objects, and readable attention overlays; not representative sampling. Unmodified rows from the recorded candidate set.')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    return output_dir/name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fusion-report', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--data-root', default='data/kitti')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--score-thr', type=float, default=.30)
    p.add_argument('--pedestrian', nargs='+', type=int,
                   default=[5472, 2892, 2477, 4657, 6475, 6828])
    p.add_argument('--cyclist', nargs='+', type=int,
                   default=[1707, 3678, 3313, 4092, 859, 1787])
    p.add_argument('--car', nargs='+', type=int,
                   default=[7462, 4578, 4312, 5010, 6868, 5431])
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.50)
    model, dataset, _, provenance = load_verified(args.fusion_report, args.device)
    args.point_cloud_range = model.fusion_module.point_cloud_range.cpu().tolist()
    args.grid_size = tuple(model.fusion_module.token_grid_size)
    args.heights = model.fusion_module.height_anchors.cpu().tolist()
    if tuple(model.fusion_module.bottleneck_tokens.shape[1:]) != (8, 256):
        raise ValueError('Expected the larger KITTI eight-token checkpoint')
    infos = mmengine.load(Path(args.data_root)/'kitti_infos_val.pkl')['data_list']
    indexes = {int(x['sample_idx']): i for i, x in enumerate(infos)}
    extent = (args.point_cloud_range[0], args.point_cloud_range[1],
              args.point_cloud_range[3], args.point_cloud_range[4])
    manifest = dict(
        model='ABFuse ResNet-50/FPN, eight bottleneck tokens, epoch 14',
        checkpoint=provenance,
        selection='Alternative scenes selected by input visibility and scene variety from candidate sheets. Illustrative, not representative or evidence of localized causal importance.',
        attention_method='Unchanged head-averaged final-layer attention-path product, independent display normalization and calibrated height-weighted camera projection.',
        examples=[], skipped=[], outputs={})
    groups = []
    for label, ids in enumerate([args.pedestrian, args.cyclist, args.car]):
        rows = []
        for sid in ids:
            i = indexes[sid]
            if int(Path(dataset.get_data_info(i)['lidar_path']).stem) != sid:
                raise ValueError('Frame identity mismatch')
            with torch.no_grad():
                fusion, pred, _ = run_model_with_attention(model, dataset, i)
                ordinary = model.test_step(pseudo_collate([dataset[i]]))[0].pred_instances_3d
            if (len(ordinary.scores_3d) != len(pred.scores_3d) or
                    not torch.allclose(ordinary.scores_3d, pred.scores_3d, atol=1e-4, rtol=1e-4) or
                    not torch.allclose(ordinary.bboxes_3d.tensor, pred.bboxes_3d.tensor, atol=1e-3, rtol=1e-4)):
                raise ValueError('Attention capture differs from ordinary inference')
            if model.training or not fusion['lidar_token_keep_mask'].all():
                raise ValueError('Training masking active')
            try:
                row, record = attention_example(infos[i], label, fusion, pred, args, extent)
            except ValueError as error:
                if 'true positive' not in str(error) and 'No moderate' not in str(error):
                    raise
                manifest['skipped'].append(dict(sample_idx=sid, reason=str(error)))
                print(f'Skipped {sid:06d}: {error}', flush=True)
                continue
            filename = f'{CLASS_NAMES[label].lower()}_{sid:06d}.png'
            if not cv2.imwrite(str(args.output_dir/filename), row):
                raise RuntimeError(filename)
            record.update(filename=filename, attention_inference_matches_standard=True)
            manifest['examples'].append(record)
            rows.append((row, record))
            print(f'Rendered {CLASS_NAMES[label]} {sid:06d}; 3D IoU={record["matched_3d_iou"]:.3f}', flush=True)
        if not rows:
            raise RuntimeError(f'No usable {CLASS_NAMES[label]} candidates')
        groups.append(rows)
        thumbs = [cv2.resize(x[0], (1050, 390), interpolation=cv2.INTER_AREA) for x in rows]
        name = f'{CLASS_NAMES[label].lower()}_alternatives.png'
        cv2.imwrite(str(args.output_dir/name), stack(thumbs))
    for index in range(min(3, min(map(len, groups)))):
        name = f'attention_maps_camera_bev_option_{index+1}.png'
        cv2.imwrite(str(args.output_dir/name), stack([g[index][0] for g in groups]))
        manifest['outputs'][name] = dict(
            samples=[g[index][1]['sample_idx'] for g in groups],
            sha256=sha256(args.output_dir/name))
    (args.output_dir/'attention_alternatives_manifest.json').write_text(
        json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
