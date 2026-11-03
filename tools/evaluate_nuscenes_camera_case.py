"""Strict, full-validation camera corruption of a trained nuScenes detector.

No training hooks, optimizer, weight conversion, or architecture changes are
allowed. Black means zero RGB/BGR pixels BEFORE normalization, not zero features.
Shuffling replaces the complete six-camera bundle while retaining target-scene
LiDAR, annotations, and calibration. This is an OOD dependence test.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner, set_random_seed
from mmengine.utils import import_modules_from_strings
from mmdet3d.utils import register_all_modules


CORRUPTIONS = {'BlackImage', 'ShuffleNuScenesImagePaths'}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def camera_pipeline(pipeline, mode, info_file, seed=0):
    result = deepcopy(pipeline)
    if any(step['type'] in CORRUPTIONS for step in result):
        raise ValueError('Use a clean source config, not an ablation config')
    if mode not in ('correct', 'black', 'shuffled'):
        raise ValueError(mode)
    load = next((i for i, step in enumerate(result)
                 if step['type'].split('.')[-1] == 'LoadMultiViewImageFromFiles'), None)
    if load is None:
        raise ValueError('Six-camera image loader not found')
    if mode == 'shuffled':
        result.insert(load, dict(type='ShuffleNuScenesImagePaths',
                                info_file=str(info_file), seed=seed))
    elif mode == 'black':
        index = load + 1
        while index < len(result) and result[index]['type'] in ('MultiViewWrapper', 'Resize'):
            index += 1
        result.insert(index, dict(type='BlackImage'))
    return result


def check_camera_inputs(dataset, clean_pipeline, info_file):
    """Check actual pixels and unchanged points/calibration on one real sample."""
    info = dataset.get_data_info(0)
    reference = Compose(clean_pipeline)(deepcopy(info))
    reference_meta = reference['data_samples'].metainfo
    assert torch.count_nonzero(reference['inputs']['img']) > 0
    for mode in ('black', 'shuffled'):
        example = Compose(camera_pipeline(clean_pipeline, mode, info_file))(deepcopy(info))
        meta = example['data_samples'].metainfo
        assert torch.equal(reference['inputs']['points'], example['inputs']['points'])
        assert meta['sample_idx'] == reference_meta['sample_idx']
        for key in ('cam2img', 'lidar2cam', 'img_shape'):
            np.testing.assert_array_equal(np.asarray(meta[key]), np.asarray(reference_meta[key]))
        if mode == 'black':
            assert torch.count_nonzero(example['inputs']['img']) == 0
            assert meta['img_path'] == reference_meta['img_path']
        else:
            assert len(meta['img_path']) == 6
            assert all(a != b and Path(a).parent == Path(b).parent
                       for a, b in zip(meta['img_path'], reference_meta['img_path']))
    return dict(actual_black_pixels_zero=True, six_shuffled_paths_changed=True,
                same_lidar_and_calibration=True, shuffle_seed=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=('correct', 'black', 'shuffled'), required=True)
    parser.add_argument('--work-dir', required=True)
    args = parser.parse_args()
    output = Path(args.work_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(status='running', mode=args.mode,
                  source_config=str(Path(args.config).resolve()),
                  checkpoint=str(Path(args.checkpoint).resolve()),
                  checkpoint_sha256=digest(args.checkpoint), optimizer_updates=0)
    handle = None
    try:
        torch.set_num_threads(4)
        set_random_seed(0)
        register_all_modules(init_default_scope=True)
        cfg = Config.fromfile(args.config)
        imports = list(cfg.get('custom_imports', {}).get('imports', []))
        name = 'projects.myfusion.transforms.image_ablation'
        if name not in imports:
            imports.append(name)
        cfg.custom_imports = dict(imports=imports, allow_failed_imports=False)
        import_modules_from_strings(**cfg.custom_imports)
        clean = deepcopy(cfg.test_dataloader.dataset.pipeline)
        if cfg.test_dataloader.dataset.get('indices') is not None:
            raise ValueError('Full validation is required, not a subset')
        if cfg.test_dataloader.sampler.get('shuffle', False):
            raise ValueError('Validation sample order must not be shuffled')
        for step in clean:
            if step['type'].split('.')[-1] == 'LoadPointsFromMultiSweeps' and not step.get('test_mode', False):
                raise ValueError('Sweep selection must already be deterministic in the source config')
        info_file = cfg.test_evaluator.ann_file
        report['annotation_sha256'] = digest(info_file)
        cfg.test_dataloader.dataset.pipeline = camera_pipeline(clean, args.mode, info_file)
        cfg.test_dataloader.update(batch_size=1, num_workers=2,
                                   persistent_workers=False, pin_memory=False)
        cfg.work_dir = str(output)
        cfg.load_from = args.checkpoint
        cfg.resume = False
        cfg.launcher = 'none'
        cfg.custom_hooks = []
        for key in ('train_cfg', 'train_dataloader', 'optim_wrapper', 'param_scheduler',
                    'val_cfg', 'val_dataloader', 'val_evaluator'):
            cfg[key] = None
        cfg.test_evaluator.jsonfile_prefix = str(output / 'results')
        cfg.test_evaluator.format_only = False
        cfg.randomness = dict(seed=0, deterministic=False)
        cfg.default_hooks.logger.interval = 250
        runner = Runner.from_cfg(cfg)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        runner.model.load_state_dict(checkpoint['state_dict'], strict=True)
        report['strict_checkpoint_load_passed'] = True
        del checkpoint
        dataset = runner.test_dataloader.dataset
        if len(dataset) != 6019:
            raise RuntimeError(f'Expected 6019 validation frames, found {len(dataset)}')
        report['validation_samples'] = len(dataset)
        report['input_preflight'] = check_camera_inputs(dataset, clean, info_file)
        gc.collect()
        checked = [0]

        def check_eval(module, inputs, result):
            if module.training:
                raise RuntimeError('Fusion is not in eval mode')
            if checked[0] % 250 == 0:
                if not result['lidar_token_keep_mask'].all():
                    raise RuntimeError('LiDAR masking active at evaluation')
                if not torch.isfinite(result['bev_features']).all():
                    raise RuntimeError('Non-finite BEV features')
            checked[0] += 1

        handle = runner.model.fusion_module.register_forward_hook(check_eval)
        (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        runner.test()
        summary_path = output / 'results/pred_instances_3d/metrics_summary.json'
        summary = json.loads(summary_path.read_text())
        predictions = output / 'results/pred_instances_3d/results_nusc.json'
        tokens = set(json.loads(predictions.read_text())['results'])
        expected_tokens = {dataset.get_data_info(i)['token'] for i in range(len(dataset))}
        if len(tokens) != 6019 or tokens != expected_tokens:
            raise RuntimeError('Persisted prediction tokens do not match the validation split')
        values = dict(mAP=summary['mean_ap'], NDS=summary['nd_score'])
        if not all(np.isfinite(value) for value in values.values()):
            raise RuntimeError('Non-finite aggregate metric')
        if digest(args.checkpoint) != report['checkpoint_sha256']:
            raise RuntimeError('Checkpoint file changed during evaluation')
        report.update(status='passed', metrics=values,
                      metrics_summary=str(summary_path), predictions=str(predictions),
                      fusion_eval_batches_checked=checked[0])
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if handle is not None:
            handle.remove()
        report['elapsed_seconds'] = time.monotonic() - started
        (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print('CAMERA EVALUATION PASSED', args.mode, report['metrics'], flush=True)


if __name__ == '__main__':
    main()
