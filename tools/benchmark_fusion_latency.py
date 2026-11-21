"""Measure matched KITTI fusion and detector latency on the same real frames."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules

from projects.myfusion.fusion.simple_bev_fusion import SimpleBEVFusion
from projects.myfusion.fusion.full_attention_bev import FullAttentionBEVFusion
from projects.myfusion.tools.measure_gated_study_compute import Fusion, analyze
from projects.myfusion.tools.run_gated_studies import sha, write


ROOT = Path('/home/yab/mmdetection3d')
REFERENCE = ROOT / 'work_dirs/part_b_kitti_gated_full_20261109'


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000, result


def summary(values):
    return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                p10_ms=float(np.percentile(values, 10)),
                p90_ms=float(np.percentile(values, 90)), calls=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, default=REFERENCE,
                        help='Gated epoch-18 checkpoint and gated_resolved_config.py')
    parser.add_argument('--full-attention-dir', type=Path,
                        default=ROOT / 'work_dirs/part_b_run4_full_attention_20261109',
                        help='Full-attention epoch-16 checkpoint and run4_resolved_config.py')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.reference_dir / 'gated_resolved_config.py')
    cfg.model.img_backbone.init_cfg = None
    import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model).cuda().eval()
    checkpoint, = args.reference_dir.glob('best_*epoch_18.pth')
    assert sha(checkpoint) == 'c88b971c894716238659d5682ef9dc0801288d590d391a2d68d00fd2388a5ca0'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    indices = np.linspace(0, len(dataset) - 1, 10, dtype=int).tolist()
    # Load/resize outside timing. Each detector call preprocesses a fresh copy
    # because the legacy pillar encoder mutates its voxel input in place.
    batches = [pseudo_collate([dataset[i]]) for i in indices]
    original = model.fusion_module
    controls = {'gated_b8': original}
    average = MODELS.build(cfg.model.fusion_module).cuda().eval()
    average.load_state_dict(original.state_dict(), strict=True)
    average.use_bottleneck_gate = False
    controls['average_b8'] = average
    full_cfg = Config.fromfile(args.full_attention_dir / 'run4_resolved_config.py')
    full = MODELS.build(full_cfg.model.fusion_module).cuda().eval()
    full_checkpoint, = args.full_attention_dir.glob('best_*epoch_16.pth')
    assert sha(full_checkpoint) == 'ed20e87154879cd5ac10683437becf5f03f589897c9a3c80ff4cd46704c87dcd'
    full_state = torch.load(full_checkpoint, map_location='cpu')['state_dict']
    full.load_state_dict({k.removeprefix('fusion_module.'): v for k, v in full_state.items()
                          if k.startswith('fusion_module.')}, strict=True)
    controls['full_attention'] = full
    for combination in ('concat', 'add'):
        candidate = deepcopy(cfg.model.fusion_module)
        candidate.update(type='SimpleBEVFusion', combination=combination)
        controls[combination] = MODELS.build(candidate).cuda().eval()
    report = dict(
        status='running', gpu=torch.cuda.get_device_name(), pytorch=torch.__version__,
        batch_size=1, frame_indices=indices, warmup_calls=10,
        detector_scope='Model preprocessing/voxelization, encoders, fusion, detection head and postprocessing; excludes disk loading and dataset transforms.',
        fusion_scope='Cached real camera/LiDAR features plus geometric sampling and fusion.',
        note='Concat/add controls have fresh fusion weights. Timings compare inference graphs; detection accuracy is not compared. All models share sensor encoders and the frozen detection head.',
        checkpoint_sha256=sha(checkpoint), full_attention_checkpoint_sha256=sha(full_checkpoint),
        arms={})
    try:
        with torch.no_grad():
            processed = model.data_preprocessor(deepcopy(batches[0]), training=False)
            image = model.extract_img_feat(processed['inputs']['imgs'])
            lidar = model.extract_pts_feat(processed['inputs']['voxels'], 1)
            metadata = model._projection_metadata(processed['data_samples'], processed['inputs']['imgs'])
            for name, fusion in controls.items():
                wrapper = Fusion(fusion, metadata).eval()
                record = dict(fusion_parameters=sum(p.numel() for p in fusion.parameters()),
                              **analyze(wrapper, (image, lidar)))
                for _ in range(10):
                    wrapper(image, lidar)
                durations = [timed(lambda: wrapper(image, lidar))[0] for _ in range(100)]
                record['fusion_latency'] = summary(durations)
                report['arms'][name] = record
                write(args.output, report)
            durations = {name: [] for name in controls}
            for name, fusion in controls.items():
                model.fusion_module = fusion
                for _ in range(10):
                    batch = model.data_preprocessor(deepcopy(batches[0]), training=False)
                    model(**batch, mode='predict')
            generator = np.random.default_rng(0)
            for repeat in range(5):
                for batch in batches:
                    for name in generator.permutation(list(controls)):
                        model.fusion_module = controls[name]
                        fresh = deepcopy(batch)

                        def predict():
                            inputs = model.data_preprocessor(fresh, training=False)
                            return model(**inputs, mode='predict')

                        latency, _ = timed(predict)
                        durations[name].append(latency)
            for name, values in durations.items():
                report['arms'][name]['detector_latency'] = summary(values)
            gated = report['arms']['gated_b8']
            for values in report['arms'].values():
                values['fusion_speedup_over_gated'] = gated['fusion_latency']['median_ms'] / values['fusion_latency']['median_ms']
                values['detector_speedup_over_gated'] = gated['detector_latency']['median_ms'] / values['detector_latency']['median_ms']
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
