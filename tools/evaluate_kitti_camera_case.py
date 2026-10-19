"""One full-validation KITTI camera ablation with strict checkpoint loading.

Run cases in separate processes to release GPU memory between evaluations.
Use batch one, few workers, and a memory cap when training concurrently.
This does not update weights or modify the original dataset/config files.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from mmengine import load
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings


def camera_pipeline(pipeline, mode, split_file, seed=0):
    pipeline = copy.deepcopy(pipeline)
    if mode == 'correct':
        return pipeline
    image_load = next((index for index, step in enumerate(pipeline)
                       if step['type'].endswith('LoadImageFromFile')), None)
    if image_load is None:
        raise ValueError('Camera corruption requires an image-loading pipeline')
    if any(step['type'] in ('BlackImage', 'ShuffleKittiImagePath')
           for step in pipeline):
        raise ValueError('Use a clean source config, not an ablation config')
    if mode == 'shuffled':
        pipeline.insert(image_load, dict(
            type='ShuffleKittiImagePath', split_file=split_file, seed=seed))
    elif mode == 'black':
        # After resizing but before packing/normalization, matching the
        # repository's existing black-image protocol.
        index = image_load + 1
        while index < len(pipeline) and pipeline[index]['type'] == 'Resize':
            index += 1
        pipeline.insert(index, dict(type='BlackImage'))
    else:
        raise ValueError(f'Unknown camera mode: {mode}')
    return pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=('correct', 'black', 'shuffled'),
                        required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--memory-fraction', type=float, default=0.20)
    parser.add_argument('--split-file', default='data/kitti/ImageSets/val.txt')
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1:
        parser.error('--memory-fraction must be in (0, 1]')
    output = Path(args.work_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new evaluation directory')
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = dict(status='running', mode=args.mode,
                  checkpoint=str(Path(args.checkpoint).resolve()),
                  source_config=str(Path(args.config).resolve()))
    try:
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.cuda.reset_peak_memory_stats()
        cfg = Config.fromfile(args.config)
        imports = list(cfg.get('custom_imports', {}).get('imports', []))
        ablation_import = 'projects.myfusion.transforms.image_ablation'
        if ablation_import not in imports:
            imports.append(ablation_import)
        cfg.custom_imports = dict(imports=imports, allow_failed_imports=False)
        import_modules_from_strings(**cfg.custom_imports)
        pipeline = camera_pipeline(
            cfg.test_dataloader.dataset.pipeline, args.mode, args.split_file)
        if cfg.test_dataloader.dataset.get('indices') is not None:
            raise ValueError('Do not use a subset for full camera ablations')
        cfg.test_dataloader.dataset.pipeline = pipeline
        cfg.test_pipeline = pipeline
        cfg.test_dataloader.batch_size = 1
        cfg.test_dataloader.num_workers = 1
        cfg.test_dataloader.persistent_workers = False
        cfg.test_dataloader.pin_memory = False
        cfg.work_dir = str(output)
        cfg.load_from = args.checkpoint
        cfg.resume = False
        cfg.launcher = 'none'
        cfg.custom_hooks = []
        cfg.train_cfg = None
        cfg.train_dataloader = None
        cfg.optim_wrapper = None
        cfg.param_scheduler = None
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None
        cfg.test_evaluator.pklfile_prefix = str(output / 'predictions')
        cfg.default_hooks.logger.interval = 250
        cfg.randomness = dict(seed=0, deterministic=False)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        report['checkpoint_sha256'] = hashlib.sha256(
            Path(args.checkpoint).read_bytes()).hexdigest()
        runner = Runner.from_cfg(cfg)
        # Do not silently accept an architecture/checkpoint mismatch. This
        # also exercises the historical PointPillars-prefix remapping.
        runner.model.load_state_dict(checkpoint['state_dict'], strict=True)
        del checkpoint
        report['strict_checkpoint_load_passed'] = True
        samples = len(runner.test_dataloader.dataset)
        annotations = len(load(cfg.test_evaluator.ann_file)['data_list'])
        if samples != 3769 or samples != annotations:
            raise RuntimeError(f'Unexpected KITTI split sizes: {samples}, '
                               f'{annotations}; expected 3769 validation frames')
        report['validation_samples'] = samples
        report['gpu'] = torch.cuda.get_device_name()
        report['memory_fraction'] = args.memory_fraction
        sampled_checks = [0]

        def check_eval(module, _inputs, result):
            if module.training:
                raise RuntimeError('Fusion module is not in eval mode')
            if sampled_checks[0] % 250 == 0:
                if not result['lidar_token_keep_mask'].all():
                    raise RuntimeError('LiDAR masking was active during evaluation')
                if not torch.isfinite(result['bev_features']).all():
                    raise RuntimeError('Non-finite evaluation features')
            sampled_checks[0] += 1

        handle = None
        if hasattr(runner.model, 'fusion_module'):
            handle = runner.model.fusion_module.register_forward_hook(check_eval)
        try:
            metrics = runner.test()
        finally:
            if handle is not None:
                handle.remove()
        report['metrics'] = {name: float(value) for name, value in metrics.items()}
        # KittiMetric persists formatted KITTI predictions for reanalysis.
        predictions = output / 'predictions/pred_instances_3d.pkl'
        if not predictions.is_file():
            raise FileNotFoundError(predictions)
        if len(load(str(predictions))) != samples:
            raise RuntimeError('Persisted prediction count differs from validation')
        report['predictions'] = str(predictions)
        report['fusion_eval_batches_checked'] = sampled_checks[0]
        report['peak_gpu_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        report['elapsed_seconds'] = time.monotonic() - start
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (output / 'metrics.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('CAMERA EVALUATION PASSED:', args.mode, output, flush=True)


if __name__ == '__main__':
    main()
