"""Plan, preflight, train, and evaluate matched KITTI reliability pilots.

Use an isolated projects/myfusion source tree and a unique --root directory.
No original checkpoint or dataset is modified. Queue mode resumes completed
stages and accumulation-aligned native training checkpoints after shutdown.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from mmengine import load
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules

from projects.myfusion.tools.run_gated_studies import sha, write


ROOT = Path('/home/yab/mmdetection3d')
REFERENCE = ROOT / 'work_dirs/part_b_kitti_gated_full_20261109'
REFERENCE_SHA = 'c88b971c894716238659d5682ef9dc0801288d590d391a2d68d00fd2388a5ca0'
FROZEN = ('img_backbone', 'img_neck', 'voxel_encoder', 'middle_encoder',
          'backbone', 'neck', 'bbox_head')
EPOCHS = 6
ACCUMULATION = 8
MICROBATCHES = 1856
ARMS = ('guided', 'unguided')
CORRUPTIONS = ('camera_dim', 'camera_blur', 'lidar_dropout', 'lidar_noise')


def pipeline_with_corruption(pipeline, condition, severity=0):
    pipeline = deepcopy(pipeline)
    if any(step['type'] == 'SensorReliabilityCorruption' for step in pipeline):
        raise ValueError('Expected a clean source pipeline')
    pack = pipeline[-1]
    assert pack['type'] == 'Pack3DDetInputs'
    keys = tuple(pack.get('meta_keys', (
        'img_path', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img',
        'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
        'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
        'num_pts_feats', 'pcd_trans', 'sample_idx', 'pcd_scale_factor',
        'pcd_rotation', 'pcd_rotation_angle', 'lidar_path', 'transformation_3d_flow',
        'trans_mat', 'affine_aug')))
    pack['meta_keys'] = keys + ('modality_quality_target', 'sensor_condition',
                                'sensor_severity')
    pipeline.insert(len(pipeline) - 1, dict(
        type='SensorReliabilityCorruption', condition=condition, severity=severity))
    return pipeline


def plan(root, reference=REFERENCE):
    recipe_path = root / 'recipe.json'
    if recipe_path.exists():
        recipe = json.loads(recipe_path.read_text())
        assert sha(root / 'initializer.pth') == recipe['initializer_sha256']
        for arm in ARMS:
            assert sha(root / f'{arm}.py') == recipe['configs'][arm]
        return recipe
    root.mkdir(parents=True, exist_ok=False)
    checkpoint, = reference.glob('best_*epoch_18.pth')
    assert sha(checkpoint) == REFERENCE_SHA
    cfg = Config.fromfile(reference / 'gated_resolved_config.py')
    cfg.model.img_backbone.init_cfg = None
    cfg.model.type = 'ReliabilityMBTDetector'
    cfg.model.reliability_loss_weight = .1
    cfg.model.fusion_module.type = 'ReliabilityGatedMBTBEVFusion'
    cfg.model.fusion_module.camera_drop_prob = 0
    cfg.model.fusion_module.reliability_strength = 1.0
    cfg.custom_imports.imports += [
        'projects.myfusion.fusion.mbt_bev_reliability',
        'projects.myfusion.fusion.mbt_detector_reliability',
        'projects.myfusion.transforms.sensor_reliability']
    import_modules_from_strings(**cfg.custom_imports)
    torch.manual_seed(0)
    model = MODELS.build(cfg.model)
    weights = torch.load(checkpoint, map_location='cpu')['state_dict']
    state = model.state_dict()
    added = state.keys() - weights.keys()
    assert added and all(key.startswith((
        'fusion_module.camera_reliability.', 'fusion_module.lidar_reliability.'))
        for key in added)
    assert not weights.keys() - state.keys()
    state.update(weights)
    model.load_state_dict(state, strict=True)
    torch.save(dict(state_dict=state), root / 'initializer.pth')
    del model, state, weights
    cfg.load_from = str(root / 'initializer.pth')
    cfg.resume = False
    cfg.launcher = 'none'
    cfg.randomness = dict(seed=0, deterministic=False)
    cfg.train_dataloader.batch_size = 4
    cfg.train_dataloader.num_workers = 2
    cfg.train_dataloader.drop_last = True
    dataset = cfg.train_dataloader.dataset
    assert dataset.type == 'RepeatDataset' and dataset.times == 2
    dataset.dataset.pipeline = pipeline_with_corruption(dataset.dataset.pipeline, 'random')
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=EPOCHS, val_interval=2)
    cfg.optim_wrapper = dict(
        type='OptimWrapper', optimizer=dict(type='AdamW', lr=1e-4, weight_decay=.01),
        accumulative_counts=ACCUMULATION, clip_grad=dict(max_norm=10, norm_type=2))
    cfg.param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True, begin=0,
                               end=EPOCHS, T_max=EPOCHS, eta_min=1e-5,
                               convert_to_iter_based=True)]
    cfg.custom_hooks = [dict(type='StagedFusionTrainingHook', freeze_epochs=EPOCHS,
                             module_names=FROZEN)]
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', by_epoch=True,
                                        interval=1, save_last=True, max_keep_ckpts=2)
    cfg.default_hooks.logger.interval = 100
    for field in ('val_dataloader', 'test_dataloader'):
        cfg[field].batch_size = 1
        cfg[field].num_workers = 1
        cfg[field].persistent_workers = False
    hashes = {}
    for arm in ARMS:
        candidate = deepcopy(cfg)
        candidate.model.fusion_module.reliability_strength = 1.0 if arm == 'guided' else 0.0
        candidate.work_dir = str(root / arm)
        candidate.val_evaluator.pklfile_prefix = str(root / arm / 'clean_validation')
        candidate.test_evaluator.pklfile_prefix = str(root / arm / 'test_predictions')
        candidate.dump(root / f'{arm}.py')
        hashes[arm] = sha(root / f'{arm}.py')
    recipe = dict(
        status='planned', reference_checkpoint=str(checkpoint),
        reference_sha256=REFERENCE_SHA, initializer_sha256=sha(root / 'initializer.pth'),
        configs=hashes, epochs=EPOCHS, microbatches_per_epoch=MICROBATCHES,
        physical_batch=4, accumulation=ACCUMULATION, seed=0,
        frozen_modules=FROZEN, selection='Fixed final epoch 6 for both pilot arms',
        quality_targets='[1-severity,1] for camera; [1,1-severity] for LiDAR',
        corruption_training='25% clean; otherwise uniform type and Beta(2,2) severity',
        evaluation_severities=[.25, .5, .75],
        note='Scores estimate synthetic severity. Complete LiDAR absence is outside this pilot.')
    write(recipe_path, recipe)
    return recipe


class ReliabilityAudit(Hook):
    priority = 'LOWEST'

    def __init__(self, report, report_path, initializer):
        self.report, self.report_path = report, report_path
        prefixes = tuple(name + '.' for name in FROZEN)
        self.protected = {k: v for k, v in initializer.items() if k.startswith(prefixes)}

    def verify(self, model):
        for name in FROZEN:
            module = getattr(model, name)
            assert not any(p.requires_grad for p in module.parameters()), name
            assert not any(child.training for child in module.modules()), name
        for name, tensor in self.protected.items():
            assert torch.equal(model.state_dict()[name].cpu(), tensor), name

    def before_train(self, runner):
        for name in FROZEN:
            getattr(runner.model, name).requires_grad_(False).eval()
        self.verify(runner.model)
        self.start, self.steps = runner.iter, 0
        assert self.start % ACCUMULATION == 0
        self.handle = runner.optim_wrapper.optimizer.register_step_post_hook(
            lambda *args: setattr(self, 'steps', self.steps + 1))
        self.report.update(status='training', start_microbatch=self.start,
                           planned_microbatches=runner.max_iters,
                           protected_tensors=len(self.protected))
        write(self.report_path, self.report)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        loss = float(torch.as_tensor(outputs['loss']).detach())
        assert np.isfinite(loss)
        if (runner.iter + 1) % 100 == 0 or runner.iter < self.start + ACCUMULATION:
            self.report.update(completed_microbatches=runner.iter + 1,
                               optimizer_updates=self.start // ACCUMULATION + self.steps,
                               loss=loss, updated_at=datetime.now(timezone.utc).isoformat())
            write(self.report_path, self.report)

    def after_train_epoch(self, runner):
        self.verify(runner.model)

    def after_train(self, runner):
        self.verify(runner.model)
        assert self.steps == (runner.max_iters - self.start) // ACCUMULATION
        self.handle.remove()
        self.report.update(completed_microbatches=runner.max_iters,
                           optimizer_updates=runner.max_iters // ACCUMULATION,
                           frozen_sensors_and_head_unchanged=True)


def train(root, arm, smoke, memory_fraction):
    recipe = json.loads((root / 'recipe.json').read_text())
    assert sha(root / f'{arm}.py') == recipe['configs'][arm]
    assert sha(root / 'initializer.pth') == recipe['initializer_sha256']
    cfg = Config.fromfile(root / f'{arm}.py')
    output = root / ('preflight_' + arm if smoke else arm)
    progress = output / 'progress.json'
    if progress.exists() and json.loads(progress.read_text()).get('status') == 'complete':
        return
    if smoke:
        output.mkdir(exist_ok=False)
        cfg.train_dataloader.dataset.dataset.indices = 64
        cfg.train_dataloader.num_workers = 0
        cfg.train_dataloader.persistent_workers = False
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=16, val_interval=1000)
        cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
        cfg.default_hooks.checkpoint = dict(type='CheckpointHook', by_epoch=False,
                                            interval=16, max_keep_ckpts=1)
    else:
        preflight = json.loads((root / ('preflight_' + arm) / 'progress.json').read_text())
        assert preflight['status'] == 'complete'
        assert preflight['config_sha256'] == recipe['configs'][arm]
        output.mkdir(exist_ok=True)
        from projects.myfusion.tools.finetune_kitti_camera_quality import recovery_checkpoint
        # The pilot shares the same 1856-microbatch, accumulation-eight recipe.
        checkpoint = recovery_checkpoint(output)
        if checkpoint:
            cfg.resume, cfg.load_from = True, str(checkpoint)
        elif progress.exists():
            # A crash before epoch one can restart from its verified initializer.
            if list(output.glob('epoch_*.pth')):
                raise RuntimeError('Existing recovery checkpoints failed integrity checks')
    cfg.work_dir = str(output)
    report = dict(status='starting', arm=arm, smoke=smoke,
                  config_sha256=recipe['configs'][arm], resumed=bool(cfg.resume))
    write(progress, report)
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    try:
        runner = Runner.from_cfg(cfg)
        initializer = torch.load(root / 'initializer.pth', map_location='cpu')['state_dict']
        runner.model.load_state_dict(
            torch.load(cfg.load_from, map_location='cpu')['state_dict'], strict=True)
        if not smoke:
            assert len(runner.train_dataloader) == MICROBATCHES
        runner.register_hook(ReliabilityAudit(report, progress, initializer))
        runner.train()
        if smoke:
            checkpoint = output / 'iter_16.pth'
            saved = torch.load(checkpoint, map_location='cpu')
            assert saved['optimizer']['state'] and saved['param_schedulers']
            runner.model.load_state_dict(saved['state_dict'], strict=True)
            # A real annotated example probes the entire detector loss path.
            batch = next(iter(runner.train_dataloader))
            processed = runner.model.data_preprocessor(batch, training=False)
            runner.model.eval()
            runner.model.zero_grad(set_to_none=True)
            losses = runner.model(**processed, mode='loss')
            loss, _ = runner.model.parse_losses(losses)
            loss.backward()
            fusion = runner.model.fusion_module
            for module in (fusion.camera_reliability, fusion.lidar_reliability,
                           fusion.bottleneck_gates):
                gradients = [p.grad for p in module.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                assert any(g.abs().max() > 0 for g in gradients)
            fusion.camera_mode = 'missing'
            observed = []
            handle = fusion.register_forward_hook(
                lambda module, inputs, result: observed.append(
                    torch.equal(inputs[1], result['bev_features'])))
            try:
                batch = next(iter(runner.train_dataloader))
                processed = runner.model.data_preprocessor(batch, training=False)
                with torch.no_grad():
                    runner.model(**processed, mode='predict')
            finally:
                handle.remove()
            assert observed and all(observed)
            report.update(strict_reload=True, finite_quality_and_gate_gradients=True,
                          exact_explicit_missing_identity=True)
        else:
            checkpoint = output / f'epoch_{EPOCHS}.pth'
        report.update(status='complete', checkpoint=str(checkpoint),
                      checkpoint_sha256=sha(checkpoint))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        write(progress, report)


def evaluate(root, arm, condition, severity, memory_fraction):
    output = root / 'evaluations' / arm / f'{condition}_{severity:.2f}'
    report_path = output / 'metrics.json'
    if report_path.exists() and json.loads(report_path.read_text()).get('status') == 'complete':
        return
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(root / ('guided.py' if arm in ('reference', 'lidar') else f'{arm}.py'))
    cfg.work_dir, cfg.launcher, cfg.resume = str(output), 'none', False
    if arm in ARMS:
        checkpoint = root / arm / f'epoch_{EPOCHS}.pth'
        trained = json.loads((root / arm / 'progress.json').read_text())
        assert trained['status'] == 'complete'
        assert sha(checkpoint) == trained['checkpoint_sha256']
    else:
        checkpoint = root / 'initializer.pth'
        cfg.model.fusion_module.reliability_strength = 0
    if arm == 'lidar':
        cfg.model.fusion_module.camera_mode = 'missing'
    data = cfg.test_dataloader.dataset
    data.pipeline = pipeline_with_corruption(data.pipeline, condition, severity)
    cfg.test_dataloader.num_workers = 1
    cfg.test_dataloader.persistent_workers = False
    cfg.test_evaluator.pklfile_prefix = str(output / 'predictions')
    cfg.train_cfg = cfg.train_dataloader = cfg.optim_wrapper = cfg.param_scheduler = None
    cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
    cfg.custom_hooks = []
    cfg.load_from = str(checkpoint)
    cfg.randomness = dict(seed=0, deterministic=False)
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    report = dict(status='running', arm=arm, condition=condition, severity=severity,
                  checkpoint_sha256=sha(checkpoint), validation_samples=3769)
    write(report_path, report)
    try:
        runner = Runner.from_cfg(cfg)
        runner.model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
        assert len(runner.test_dataloader.dataset) == 3769
        assert len(load(cfg.test_evaluator.ann_file)['data_list']) == 3769
        scores, identity = [], []

        def capture(module, inputs, result):
            scores.append(result['modality_quality'].detach().cpu())
            if arm == 'lidar':
                identity.append(torch.equal(inputs[1], result['bev_features']))

        handle = runner.model.fusion_module.register_forward_hook(capture)
        try:
            metrics = runner.test()
        finally:
            handle.remove()
        scores = torch.cat(scores)
        assert scores.shape == (3769, 2) and torch.isfinite(scores).all()
        assert len(load(output / 'predictions/pred_instances_3d.pkl')) == 3769
        if arm == 'lidar':
            assert len(identity) == 3769 and all(identity)
        target = torch.ones_like(scores)
        if condition.startswith('camera_'):
            target[:, 0] = 1 - severity
        if condition.startswith('lidar_'):
            target[:, 1] = 1 - severity
        report.update(status='complete', metrics={k: float(v) for k, v in metrics.items()},
                      mean_quality=scores.mean(0).tolist(),
                      quality_mae=(scores - target).abs().mean(0).tolist(),
                      exact_lidar_feature_identity=all(identity) if arm == 'lidar' else None)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        write(report_path, report)


def queue(root, memory_fraction):
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stages = [('preflight', arm) for arm in ARMS] + [('train', arm) for arm in ARMS]
    state = dict(status='running', root=str(root))
    try:
        for mode, arm in stages:
            state.update(stage=mode, arm=arm)
            write(root / 'queue_status.json', state)
            subprocess.run([sys.executable, __file__, mode, '--root', str(root),
                            '--arm', arm, '--memory-fraction', str(memory_fraction)], check=True)
        cases = [('clean', 0)] + [(c, s) for c in CORRUPTIONS for s in (.25, .5, .75)]
        for condition, severity in cases:
            # Camera corruption does not change the LiDAR-only input.
            arms = ('reference', *ARMS, 'lidar') if not condition.startswith('camera_') else ('reference', *ARMS)
            for arm in arms:
                state.update(stage='evaluate', arm=arm, condition=condition, severity=severity)
                write(root / 'queue_status.json', state)
                subprocess.run([
                    sys.executable, __file__, 'evaluate', '--root', str(root), '--arm', arm,
                    '--condition', condition, '--severity', str(severity),
                    '--memory-fraction', str(memory_fraction)], check=True)
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        write(root / 'queue_status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'preflight', 'train', 'evaluate', 'queue'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, default=REFERENCE,
                        help='Gated epoch-18 checkpoint and gated_resolved_config.py (plan only)')
    parser.add_argument('--arm', choices=(*ARMS, 'reference', 'lidar'), default='guided')
    parser.add_argument('--condition', choices=('clean', *CORRUPTIONS), default='clean')
    parser.add_argument('--severity', type=float, default=0)
    parser.add_argument('--memory-fraction', type=float, default=.55)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1 or not 0 <= args.severity <= 1:
        parser.error('Invalid memory fraction or severity')
    torch.set_num_threads(2)
    register_all_modules(init_default_scope=True)
    root = args.root.resolve()
    if args.mode == 'plan':
        plan(root, args.reference_dir.resolve())
    elif args.mode == 'queue':
        queue(root, args.memory_fraction)
    elif args.mode == 'evaluate':
        evaluate(root, args.arm, args.condition, args.severity, args.memory_fraction)
    else:
        if args.arm not in ARMS:
            parser.error('Training requires guided or unguided arm')
        train(root, args.arm, args.mode == 'preflight', args.memory_fraction)


if __name__ == '__main__':
    main()
