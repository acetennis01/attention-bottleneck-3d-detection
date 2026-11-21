"""Fine-tune KITTI fusion and the image-conditioned residual weight."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules

from projects.myfusion.tools.run_gated_studies import sha, write


FROZEN_MODULES = (
    'img_backbone', 'img_neck', 'voxel_encoder', 'middle_encoder',
    'backbone', 'neck', 'bbox_head',
)
EPOCHS = 6
BATCH = 4
ACCUMULATION = 8
KITTI_SAMPLES = 3712
MICROBATCHES_PER_EPOCH = 2 * KITTI_SAMPLES // BATCH
FIXED_WEIGHT = 0.77


def plan(pilot_model, root):
    preflight = json.loads((pilot_model / 'preflight.json').read_text())
    checkpoint = pilot_model / 'quality_weighted.pth'
    if preflight['status'] != 'passed' or sha(checkpoint) != preflight['combined_checkpoint_sha256']:
        raise ValueError('Pilot checkpoint does not match the verified KITTI model')
    root.mkdir(parents=True, exist_ok=False)
    cfg = Config.fromfile(pilot_model / 'resolved_config.py')
    cfg.custom_imports.imports.append('projects.myfusion.transforms.kitti_quality_eval')
    assert cfg.model.type == 'KittiQualityWeightedDetector'
    assert cfg.model.fusion_module.use_bottleneck_gate
    assert cfg.model.fusion_module.bypass_missing
    cfg.model.img_backbone.init_cfg = None
    cfg.model.use_camera_quality = True
    cfg.model.fixed_camera_quality = None
    cfg.work_dir = str(root / 'train')
    cfg.resume = False
    cfg.load_from = str(root / 'initializer.pth')
    cfg.train_dataloader.batch_size = BATCH
    cfg.train_dataloader.num_workers = 2
    cfg.train_dataloader.drop_last = True
    repeat = cfg.train_dataloader.dataset
    assert repeat.type == 'RepeatDataset' and repeat.times == 2
    train_pipeline = repeat.dataset.pipeline
    resize = next(i for i, step in enumerate(train_pipeline) if step.type == 'Resize')
    train_pipeline.insert(resize + 1, dict(type='KittiRandomQualityCorruption',
                                           probability=0.5))
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=EPOCHS, val_interval=2)
    cfg.optim_wrapper = dict(
        type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),
        accumulative_counts=ACCUMULATION, clip_grad=dict(max_norm=10, norm_type=2))
    cfg.param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True,
                                begin=0, end=EPOCHS, T_max=EPOCHS,
                                eta_min=0.00001, convert_to_iter_based=True)]
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', by_epoch=True, interval=1,
                                        max_keep_ckpts=2, save_last=True)
    cfg.custom_hooks = [dict(type='StagedFusionTrainingHook', freeze_epochs=EPOCHS,
                             module_names=list(FROZEN_MODULES))]
    cfg.val_evaluator.pklfile_prefix = str(root / 'training_validation')
    weights = torch.load(checkpoint, map_location='cpu')['state_dict']
    # Retain the pilot's image encoder, but reset its saturated output to a
    # common residual scale so detection gradients can reach every condition.
    weights['camera_quality_head.layers.8.weight'].zero_()
    weights['camera_quality_head.layers.8.bias'].fill_(
        math.log(FIXED_WEIGHT / (1 - FIXED_WEIGHT)))
    torch.save(dict(state_dict=weights), cfg.load_from)
    cfg.dump(root / 'resolved_config.py')
    write(root / 'recipe.json', dict(
        pilot_checkpoint=str(checkpoint), pilot_sha256=sha(checkpoint),
        initializer_sha256=sha(cfg.load_from), epochs=EPOCHS,
        physical_batch=BATCH, accumulation=ACCUMULATION,
        fixed_weight_control=FIXED_WEIGHT, checkpoint_selection='fixed final epoch',
        config_sha256=sha(root / 'resolved_config.py'), frozen_modules=FROZEN_MODULES))


def recovery_checkpoint(output):
    candidates = []
    for path in output.glob('epoch_*.pth'):
        match = re.fullmatch(r'epoch_(\d+)\.pth', path.name)
        if match:
            candidates.append((int(match[1]), path))
    for epoch, checkpoint in sorted(candidates, reverse=True):
        try:
            saved = torch.load(checkpoint, map_location='cpu')
        except (OSError, EOFError, RuntimeError, ValueError):
            continue
        meta = saved.get('meta', {})
        if (meta.get('epoch') == epoch and meta.get('iter') == epoch * MICROBATCHES_PER_EPOCH
                and saved.get('optimizer', {}).get('state') and saved.get('param_schedulers')):
            return checkpoint
    return None


class QualityAudit(Hook):
    priority = 'LOWEST'

    def __init__(self, report, report_path, base_weights):
        prefixes = tuple(name + '.' for name in FROZEN_MODULES)
        self.protected = {name: value for name, value in base_weights.items()
                          if name.startswith(prefixes)}
        self.report = report
        self.report_path = report_path
        self.steps = 0

    def verify_frozen(self, model):
        for name in FROZEN_MODULES:
            module = getattr(model, name)
            assert not any(p.requires_grad for p in module.parameters()), name
            assert not any(child.training for child in module.modules()), name
        state = model.state_dict()
        for name, expected in self.protected.items():
            assert torch.equal(state[name].cpu(), expected), name

    def before_train(self, runner):
        for name in FROZEN_MODULES:
            getattr(runner.model, name).requires_grad_(False).eval()
        self.verify_frozen(runner.model)
        self.start = runner.iter
        self.expected = runner.max_iters
        assert self.start % ACCUMULATION == 0
        self.step_handle = runner.optim_wrapper.optimizer.register_step_post_hook(
            lambda *args: setattr(self, 'steps', self.steps + 1))
        self.report.update(status='training', start_microbatch=self.start,
                           planned_microbatches=self.expected,
                           protected_tensors=len(self.protected))
        write(self.report_path, self.report)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        loss = float(torch.as_tensor(outputs['loss']).detach())
        assert math.isfinite(loss)
        microbatch = runner.iter + 1
        if microbatch % 100 == 0 or microbatch == self.start + ACCUMULATION:
            self.report.update(completed_microbatches=microbatch,
                               optimizer_updates=self.start // ACCUMULATION + self.steps,
                               loss=loss, updated_at=datetime.now(timezone.utc).isoformat())
            write(self.report_path, self.report)

    def after_train_epoch(self, runner):
        self.verify_frozen(runner.model)

    def after_train(self, runner):
        self.verify_frozen(runner.model)
        assert self.steps == (self.expected - self.start) // ACCUMULATION
        self.step_handle.remove()
        self.report.update(completed_microbatches=self.expected,
                           optimizer_updates=self.start // ACCUMULATION + self.steps,
                           frozen_inputs_and_detector_unchanged=True)


def check_smoke(runner, cfg, validation_dataset, output, report):
    model = runner.model
    saved = torch.load(output / 'iter_16.pth', map_location='cpu')
    model.load_state_dict(saved['state_dict'], strict=True)
    assert saved['optimizer']['state'] and saved['param_schedulers']
    model.eval()

    def processed_sample():
        return model.data_preprocessor(
            pseudo_collate([validation_dataset[0]]), training=False)

    batch = processed_sample()
    model.zero_grad(set_to_none=True)
    losses = model(**batch, mode='loss')
    detection_loss, _ = model.parse_losses({
        key: value for key, value in losses.items() if key != 'loss_camera_aux'})
    detection_loss.backward()
    groups = {
        'camera quality': model.camera_quality_head.parameters(),
        'bottleneck': [model.fusion_module.bottleneck_tokens],
        'bottleneck gate': model.fusion_module.bottleneck_gates.parameters(),
    }
    for name, parameters in groups.items():
        grads = [p.grad for p in parameters if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        assert any(g.count_nonzero().item() for g in grads), name
    model.zero_grad(set_to_none=True)
    model.fusion_module.camera_mode = 'missing'
    verified = []

    def check_fallback(_module, inputs, result):
        assert torch.equal(inputs[1], result['bev_features'])
        assert not result['camera_residual_weight'].any()
        verified.append(True)

    handle = model.fusion_module.register_forward_hook(check_fallback)
    try:
        with torch.no_grad():
            predictions = model(**processed_sample(), mode='predict')
    finally:
        handle.remove()
    assert verified and all(torch.isfinite(p.pred_instances_3d.scores_3d).all()
                            for p in predictions)
    report.update(status='passed', strict_native_reload=True,
                  detection_gradient_groups=list(groups), exact_missing_identity=True,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)


def train(root, smoke, memory_fraction):
    torch.set_num_threads(2)
    if memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    register_all_modules(init_default_scope=True)
    recipe = json.loads((root / 'recipe.json').read_text())
    config = root / 'resolved_config.py'
    assert sha(config) == recipe['config_sha256']
    cfg = Config.fromfile(config)
    assert sha(cfg.load_from) == recipe['initializer_sha256']
    preflight_path = root / 'preflight' / 'progress.json'
    validation = None
    if smoke:
        output = root / 'preflight'
        output.mkdir(exist_ok=False)
        cfg.work_dir = str(output)
        # The detection gradient probe requires annotations, so use a
        # deterministic training item and disable its random corruption.
        dataset = cfg.train_dataloader.dataset.dataset
        probe = dataset.copy()
        probe.pipeline = [step for step in dataset.pipeline
                          if step.type != 'KittiRandomQualityCorruption']
        probe.indices = 64
        validation = DATASETS.build(probe)
        dataset.indices = 64
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=16, val_interval=1000)
        cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
        cfg.default_hooks.checkpoint = dict(type='CheckpointHook', by_epoch=False,
                                            interval=16, max_keep_ckpts=1)
    else:
        preflight = json.loads(preflight_path.read_text())
        assert preflight['status'] == 'passed'
        assert preflight['config_sha256'] == recipe['config_sha256']
        output = Path(cfg.work_dir)
        output.mkdir(exist_ok=True)
    report_path = output / 'progress.json'
    if report_path.exists() and json.loads(report_path.read_text()).get('status') == 'complete':
        return
    base_weights = torch.load(cfg.load_from, map_location='cpu')['state_dict']
    resume = None if smoke else recovery_checkpoint(output)
    if resume is not None:
        cfg.resume = True
        cfg.load_from = str(resume)
    report = dict(status='starting', smoke=smoke, config_sha256=recipe['config_sha256'],
                  resumed_from=str(resume) if resume else None)
    write(report_path, report)
    try:
        runner = Runner.from_cfg(cfg)
        if not smoke:
            assert len(runner.train_dataloader) == MICROBATCHES_PER_EPOCH
        native_weights = torch.load(cfg.load_from, map_location='cpu')['state_dict']
        runner.model.load_state_dict(native_weights, strict=True)
        runner.register_hook(QualityAudit(report, report_path, base_weights))
        runner.train()
        if smoke:
            check_smoke(runner, cfg, validation, output, report)
        else:
            checkpoint = output / f'epoch_{EPOCHS}.pth'
            report.update(status='complete', selected_checkpoint=str(checkpoint),
                          selected_checkpoint_sha256=sha(checkpoint))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        write(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'preflight', 'train'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot-model', type=Path)
    parser.add_argument('--memory-fraction', type=float)
    args = parser.parse_args()
    if args.mode == 'plan':
        if args.pilot_model is None:
            parser.error('--pilot-model is required for plan')
        plan(args.pilot_model.resolve(), args.root.resolve())
    else:
        train(args.root.resolve(), args.mode == 'preflight', args.memory_fraction)


if __name__ == '__main__':
    main()
