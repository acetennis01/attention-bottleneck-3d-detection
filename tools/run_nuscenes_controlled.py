"""Run sequential, equal-update nuScenes continuation controls.

Run from the MMDetection3D root. Smoke artifacts are never used as training
initializers. The queue stops on any failed arm, mismatched sample/input stream,
source change or missing full-validation result; it never extends the budget.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import torch
from mmengine.config import Config
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('joint', 'frozen', 'lower_lr', 'lower_aux', 'multiscale')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare_multiscale(reference, destination):
    """Explicit conversion: old tensors exact, cloned queries, zero adapters."""
    cfg = Config.fromfile(PROJECT / 'configs/nuscenes_controlled_multiscale.py')
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(reference, map_location='cpu')
    state = checkpoint['state_dict']
    actual = model.state_dict()
    added = set(actual) - set(state)
    allowed = ('fusion_module.pyramid_queries.', 'fusion_module.pyramid_residuals.')
    assert not set(state) - set(actual)
    assert added and all(key.startswith(allowed) for key in added)
    converted = dict(state)
    for key in added:
        if key.startswith(allowed[0]):
            converted[key] = state['fusion_module.lidar_projection.' + key.split('.', 3)[-1]].clone()
        else:
            assert torch.count_nonzero(actual[key]) == 0, key
            converted[key] = actual[key].clone()
    model.load_state_dict(converted, strict=True)
    for key, value in state.items():
        assert torch.equal(model.state_dict()[key], value), key
    torch.save(dict(state_dict=converted, meta=dict(
        model_only_initializer=True, source_checkpoint=str(reference.resolve()),
        source_sha256=sha256(reference), added_tensors=sorted(added))), destination)
    return dict(path=str(destination), sha256=sha256(destination), added_tensors=sorted(added))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'train'), required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--smoke-root', help='Required passing smoke suite for production launch')
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error('Duplicate arms are not allowed')
    root = Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new output root; existing experiments are never overwritten')
    root.mkdir(parents=True, exist_ok=True)
    register_all_modules(init_default_scope=True)
    source = {str(path.relative_to(PROJECT)): sha256(path)
              for path in sorted(PROJECT.rglob('*.py')) if 'work_dirs' not in path.parts}
    base = Config.fromfile(PROJECT / 'configs/nuscenes_controlled_base.py')
    reference = Path(base.reference_checkpoint).resolve()
    from projects.myfusion.hooks.controlled_retraining import ControlledRetrainingHook
    assert sha256(reference) == ControlledRetrainingHook.reference_sha256
    if args.mode == 'train':
        if not args.smoke_root:
            parser.error('--smoke-root is required for training')
        smoke = json.loads((Path(args.smoke_root) / 'suite.json').read_text())
        assert smoke['status'] == 'complete' and smoke['mode'] == 'smoke'
        assert set(args.arms) <= set(smoke['arms'])
        assert smoke['source_sha256'] == source, 'Source changed since passing smokes'
    suite = dict(status='preparing', mode=args.mode, arms=args.arms,
                 started_at=datetime.now().astimezone().isoformat(),
                 source_sha256=source, reference_checkpoint=str(reference),
                 reference_sha256=sha256(reference), results={},
                 microbatches_per_arm=8 if args.mode == 'smoke' else 4000,
                 optimizer_updates_per_arm=2 if args.mode == 'smoke' else 1000,
                 seed=0, full_validation=args.mode == 'train')
    write_json(root / 'suite.json', suite)
    try:
        if 'multiscale' in args.arms:
            suite['multiscale_initializer'] = prepare_multiscale(
                reference, root / 'multiscale_initializer.pth')
        suite['status'] = 'running'
        reference_stream = None
        for arm in args.arms:
            for relative, digest in source.items():
                assert sha256(PROJECT / relative) == digest, f'Source changed: {relative}'
            active = subprocess.check_output(['ps', '-eo', 'args'], text=True)
            if any('tools/train.py' in line and 'python' in line for line in active.splitlines()):
                raise RuntimeError('Another MMDetection3D trainer exists; refusing overlap')
            cfg = Config.fromfile(PROJECT / f'configs/nuscenes_controlled_{arm}.py')
            cfg.work_dir = str(root / arm)
            output = Path(cfg.work_dir)
            output.mkdir()
            if arm == 'multiscale':
                cfg.load_from = str(root / 'multiscale_initializer.pth')
            assert cfg.resume is False and cfg.param_scheduler == []
            assert cfg.train_dataloader.batch_size == 1
            assert cfg.optim_wrapper.accumulative_counts == 4
            assert cfg.model.fusion_module.lidar_spatial_drop_prob == 0
            assert cfg.train_dataloader.dataset.get('indices') is None
            assert len(cfg.class_names) == 10
            cfg.val_evaluator.jsonfile_prefix = str(output / 'results')
            cfg.test_evaluator = cfg.val_evaluator
            if args.mode == 'smoke':
                cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=8,
                                     val_interval=100000)
                cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
                cfg.test_cfg = cfg.test_dataloader = cfg.test_evaluator = None
                # Keep the full training sampler/pipeline and worker settings.
                cfg.default_hooks.checkpoint = dict(
                    type='CheckpointHook', interval=8, by_epoch=False, save_last=True)
                cfg.default_hooks.logger.interval = 1
                for hook in cfg.custom_hooks:
                    if hook.type == 'ControlledRetrainingHook':
                        hook.expected_updates = 2
                cfg.custom_hooks.append(dict(type='ControlledModelSmokeHook'))
            cfg.dump(str(output / 'run_config.py'))
            print(f'Starting {args.mode} arm {arm}: {output}', flush=True)
            with (output / 'train_console.log').open('x') as log:
                process = subprocess.Popen(
                    [sys.executable, '-u', 'tools/train.py', str(output / 'run_config.py')],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write_json(output / 'process.json', dict(
                    training_pid=process.pid, training_process_group=process.pid,
                    queue_pid=os.getpid(), arm=arm))
                suite['active_arm'] = arm
                write_json(root / 'suite.json', suite)
                status = process.wait()
            if status:
                raise RuntimeError(f'{arm} exited with status {status}; see {output}/train_console.log')
            report = json.loads((output / 'controlled_report.json').read_text())
            assert report['status'] == 'complete'
            assert report['optimizer_updates'] == suite['optimizer_updates_per_arm']
            if args.mode == 'smoke':
                model_checks = json.loads((output / 'model_smoke_report.json').read_text())
                assert model_checks['status'] == 'passed'
            stream = [report['sample_order_sha256'], report['first_input_hashes']]
            if reference_stream is None:
                reference_stream = stream
            else:
                assert stream == reference_stream, f'{arm}: training inputs/order differ from control'
            if args.mode == 'train':
                assert len(cfg.val_dataloader.dataset.get('indices', [])) == 0
                metrics = report['validation_metrics']
                assert any(key.endswith('/NDS') for key in metrics)
                assert any(key.endswith('/mAP') for key in metrics)
                assert (output / 'results/pred_instances_3d/results_nusc.json').is_file()
            suite['results'][arm] = report
            write_json(root / 'suite.json', suite)
            print(f'Completed {arm}: {report.get("validation_metrics", "execution smoke passed")}', flush=True)
        suite.update(status='complete', active_arm=None,
                     completed_at=datetime.now().astimezone().isoformat(),
                     matched_input_streams=True)
    except BaseException as error:
        suite.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(root / 'suite.json', suite)


if __name__ == '__main__':
    main()
