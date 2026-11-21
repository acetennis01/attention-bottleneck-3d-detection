"""Combine the selected quality head with the verified frozen KITTI model."""

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules

from projects.myfusion.tools.run_gated_studies import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--quality-fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-fraction', type=float, default=0.16)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1:
        parser.error('--memory-fraction must be in (0, 1]')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    register_all_modules(init_default_scope=True)
    base_sha = sha(args.base_checkpoint)
    if base_sha != 'c88b971c894716238659d5682ef9dc0801288d590d391a2d68d00fd2388a5ca0':
        raise ValueError('Expected the completed gated KITTI B8 epoch-18 reference')
    quality_path = args.quality_fit / 'best.pth'
    fit = json.loads((args.quality_fit / 'complete.json').read_text())
    cfg = Config.fromfile(args.base_config)
    cfg.custom_imports.imports.append(
        'projects.myfusion.fusion.mbt_detector_kitti_quality')
    import_modules_from_strings(**cfg.custom_imports)
    if cfg.model.fusion_module.type != 'GatedCameraMBTBEVFusion':
        raise ValueError('Expected the KITTI gated fusion implementation')
    cfg.model.type = 'KittiQualityWeightedDetector'
    cfg.work_dir = str(args.output)
    cfg.resume = False
    cfg.load_from = None
    args.output.mkdir(parents=True, exist_ok=False)
    model = MODELS.build(cfg.model)
    weights = torch.load(args.base_checkpoint, map_location='cpu')['state_dict']
    quality = torch.load(quality_path, map_location='cpu')
    prefix = 'camera_quality_head.'
    if any(key.startswith(prefix) for key in weights):
        raise ValueError('Base checkpoint already contains quality weights')
    for key, value in quality['state_dict'].items():
        weights[prefix + key] = value
    model.load_state_dict(weights, strict=True)
    model.cuda().eval()

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    if len(dataset) != 3769:
        raise ValueError('Expected the complete KITTI validation split')
    captures = []

    def capture(_module, inputs, result):
        captures.append((inputs[1], result['bev_features'],
                         result.get('camera_residual_weight')))

    handle = model.fusion_module.register_forward_hook(capture)
    try:
        with torch.no_grad():
            for use_quality, camera_mode in (
                    (True, 'present'), (False, 'present'), (True, 'missing')):
                model.use_camera_quality = use_quality
                model.fusion_module.camera_mode = camera_mode
                # Legacy PillarFeatureNet changes the input voxel coordinates
                # in place. Each comparison needs fresh preprocessed inputs.
                batch = pseudo_collate([dataset[0]])
                processed = model.data_preprocessor(batch, training=False)
                model(**processed, mode='predict')
    finally:
        handle.remove()
    lidar, weighted, weight = captures[0]
    unweighted_lidar, unweighted, _ = captures[1]
    missing_lidar, missing, missing_weight = captures[2]
    assert torch.isfinite(weighted).all() and 0 <= weight.item() <= 1
    assert torch.equal(lidar, unweighted_lidar)
    assert not torch.equal(unweighted, lidar), 'Smoke needs a nonzero fusion residual'
    torch.testing.assert_close(
        weighted, lidar + weight.reshape(1, 1, 1, 1) * (unweighted - lidar))
    assert torch.equal(missing, missing_lidar) and missing_weight.item() == 0
    cfg.dump(args.output / 'resolved_config.py')
    target = args.output / 'quality_weighted.pth'
    torch.save(dict(state_dict=weights), target)
    report = dict(status='passed', base_checkpoint_sha256=base_sha,
                  base_config_sha256=sha(args.base_config),
                  quality_checkpoint_sha256=sha(quality_path),
                  combined_checkpoint_sha256=sha(target),
                  quality_fit=fit, selection=quality['selection'],
                  strict_reload=True, real_sample_residual_scaling=True,
                  exact_missing_lidar_identity=True,
                  smoke_camera_weight=weight.item(),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    (args.output / 'preflight.json').write_text(encoded)
    print(encoded, flush=True)


if __name__ == '__main__':
    main()
