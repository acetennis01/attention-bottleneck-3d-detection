"""Short KITTI GPU checks using the recipe embedded in the paper checkpoint.

These subset runs test execution and gradients; their AP is not a paper result.
The historical fusion implementation is isolated from later nuScenes changes.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from mmengine import dump, load
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import MODELS


FROZEN_PREFIXES = ('voxel_encoder.', 'middle_encoder.', 'backbone.',
                   'neck.', 'bbox_head.')


def component(name):
    if name.startswith('img_backbone.'):
        return 'image_backbone'
    if name.startswith('fusion_module.bottleneck_tokens'):
        return 'bottleneck'
    if name.startswith('fusion_module.image_projection.'):
        return 'image_projection'
    if name.startswith('fusion_module.bev_residual.'):
        return 'bev_residual'
    if name.startswith('fusion_module.camera_aux_head.'):
        return 'camera_aux'
    if name.startswith(FROZEN_PREFIXES):
        return 'lidar_detector'
    return 'other'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class SmokeChecks(Hook):
    """Fail on incorrect warm-start, non-finite values, or dead gradients."""

    priority = 'LOW'

    def __init__(self, baseline, output):
        self.baseline = baseline
        self.output = output
        self.grad_max = {}
        self.losses = []
        self.valid_ratios = []
        self.weight_sum_minima = []
        self.weight_sum_maxima = []
        self.handles = []
        self.peak_allocated = 0

    def before_train(self, runner):
        state = runner.model.state_dict()
        source = torch.load(self.baseline, map_location='cpu')['state_dict']
        runner.model._remap_pointpillars_state_dict(source)
        keys = [key for key in state if key.startswith(FROZEN_PREFIXES)]
        assert keys and all(key in source for key in keys)
        assert all(torch.equal(state[k].cpu(), source[k]) for k in keys), (
            'LiDAR warm-start differs from the baseline checkpoint')
        self.frozen_start = {k: state[k].detach().cpu().clone() for k in keys}
        self.parameter_start = {
            name: parameter.detach().cpu().clone()
            for name, parameter in runner.model.named_parameters()
            if component(name) in ('image_backbone', 'bottleneck',
                                   'image_projection', 'bev_residual')
        }
        for name, parameter in runner.model.named_parameters():
            if parameter.requires_grad:
                self.handles.append(parameter.register_hook(
                    self.capture_gradient(name)))
        self.handles.append(runner.model.fusion_module.register_forward_hook(
            self.capture_features))

    def capture_gradient(self, name):
        def capture(gradient):
            # MMEngine resets the CUDA peak counter while writing log lines.
            # Preserve its maximum during backward, before those resets.
            self.peak_allocated = max(
                self.peak_allocated, torch.cuda.max_memory_allocated())
            assert torch.isfinite(gradient).all(), f'Non-finite gradient: {name}'
            group = component(name)
            maximum = float(gradient.detach().abs().max())
            self.grad_max[group] = max(self.grad_max.get(group, 0.), maximum)
        return capture

    def capture_features(self, module, inputs, output):
        for key in ('bev_features', 'image_tokens', 'lidar_tokens',
                    'bottleneck_tokens', 'camera_aux_logits'):
            assert torch.isfinite(output[key]).all(), f'Non-finite {key}'
        ratio = float(output['image_valid_ratio'])
        assert 0 < ratio <= 1, 'No calibrated camera cells are visible'
        self.valid_ratios.append(ratio)
        minimum = float(output['camera_weight_sum_min'])
        maximum = float(output['camera_weight_sum_max'])
        assert 0.999 <= minimum <= maximum <= 1.001, (
            f'Visible camera weights do not sum to one: {minimum}, {maximum}')
        self.weight_sum_minima.append(minimum)
        self.weight_sum_maxima.append(maximum)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        losses = {key: float(value) for key, value in outputs.items()}
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
        self.losses.append(losses)

    def after_val_epoch(self, runner, metrics=None):
        assert metrics and all(torch.isfinite(torch.tensor(v))
                               for v in metrics.values())
        self.output['validation_metrics'] = metrics

    def after_train(self, runner):
        state = runner.model.state_dict()
        assert all(torch.equal(state[k].cpu(), v)
                   for k, v in self.frozen_start.items()), (
            'Frozen LiDAR parameters or batch statistics changed')
        changed = {}
        for name, parameter in runner.model.named_parameters():
            if name in self.parameter_start:
                group = component(name)
                if not torch.equal(parameter.detach().cpu(),
                                   self.parameter_start[name]):
                    changed[group] = changed.get(group, 0) + 1
        for group in ('image_backbone', 'bottleneck', 'image_projection',
                      'bev_residual'):
            assert changed.get(group, 0) > 0, f'No parameter update in {group}'
            assert self.grad_max.get(group, 0) > 0, f'No gradient in {group}'
        self.output.update(
            train_iterations=len(self.losses),
            losses=self.losses,
            gradient_max_abs=self.grad_max,
            changed_parameter_tensors=changed,
            lidar_checkpoint_loaded_exactly=True,
            frozen_lidar_unchanged=True,
            visible_camera_ratio_min=min(self.valid_ratios),
            visible_camera_ratio_max=max(self.valid_ratios),
            camera_weight_sum_min=min(self.weight_sum_minima),
            camera_weight_sum_max=max(self.weight_sum_maxima),
            peak_gpu_allocated_mib=self.peak_allocated / 2**20,
        )
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-checkpoint', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--tokens', type=int, required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--mode', choices=('train', 'reference'), default='train')
    parser.add_argument('--aggregation-mode', choices=('legacy', 'valid_only'),
                        default='legacy',
                        help='Use valid_only to test the corrected token study.')
    args = parser.parse_args()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', mode=args.mode, tokens=args.tokens,
                  camera_aggregation_mode=args.aggregation_mode,
                  historical_fusion_commit='c2dad89',
                  note='Subset smoke test; AP is not a benchmark result.')
    try:
        checkpoint = torch.load(args.reference_checkpoint, map_location='cpu')
        cfg = Config.fromstring(checkpoint['meta']['cfg'], '.py')
        cfg.custom_imports = dict(imports=[
            'projects.myfusion.fusion.mbt_bev_kitti_v1',
            'projects.myfusion.fusion.mbt_detector',
            'projects.myfusion.hooks.staged_fusion_training',
            'projects.myfusion.hooks.fusion_diagnostics',
        ], allow_failed_imports=False)
        init_default_scope('mmdet3d')
        import_modules_from_strings(**cfg.custom_imports)
        cfg.model.fusion_module.type = 'KittiV1SymmetricMBTBEVFusion'
        cfg.model.fusion_module.camera_aggregation_mode = args.aggregation_mode
        reference_model = MODELS.build(cfg.model)
        reference_model.load_state_dict(checkpoint['state_dict'], strict=True)
        report['reference_checkpoint_strict_load'] = True
        del reference_model, checkpoint

        cfg.model.fusion_module.num_bottleneck_tokens = args.tokens
        cfg.work_dir = str(work_dir)
        cfg.resume = False
        cfg.launcher = 'none'
        cfg.env_cfg.dist_cfg.backend = 'nccl'
        cfg.env_cfg.mp_cfg.mp_start_method = 'fork'
        cfg.log_processor.by_epoch = True
        cfg.default_hooks.logger.interval = 8
        cfg.default_hooks.checkpoint = dict(
            type='CheckpointHook', interval=1, max_keep_ckpts=1,
            save_last=True, save_best=None)
        cfg.param_scheduler = []
        cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1,
                             val_interval=1)
        train_dataset = copy.deepcopy(cfg.train_dataloader.dataset.dataset)
        train_dataset.indices = 96
        cfg.train_dataloader.dataset = train_dataset
        cfg.train_dataloader.num_workers = 2
        cfg.train_dataloader.persistent_workers = False
        cfg.val_dataloader.dataset.indices = 32
        cfg.val_dataloader.num_workers = 2
        cfg.val_dataloader.persistent_workers = False
        cfg.val_evaluator.pklfile_prefix = str(work_dir / 'val_predictions')
        # KittiMetric requires one annotation for every prediction. Keep the
        # official metadata untouched and give this subset run its own file.
        val_infos = load(cfg.val_evaluator.ann_file)
        val_infos['data_list'] = val_infos['data_list'][:32]
        smoke_ann_file = work_dir / 'val_infos_smoke.pkl'
        dump(val_infos, smoke_ann_file)
        cfg.val_evaluator.ann_file = str(smoke_ann_file)
        cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
        cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
        cfg.test_evaluator.pklfile_prefix = str(work_dir / 'test_predictions')
        cfg.load_from = args.baseline_checkpoint
        cfg.custom_hooks[1]['interval'] = 8
        torch.cuda.reset_peak_memory_stats()

        if args.mode == 'reference':
            assert args.tokens == 4
            cfg.load_from = args.reference_checkpoint
            cfg.custom_hooks = []
            runner = Runner.from_cfg(cfg)
            checks = SmokeChecks(args.baseline_checkpoint, report)
            handle = runner.model.fusion_module.register_forward_hook(
                checks.capture_features)
            try:
                report['validation_metrics'] = runner.test()
            finally:
                handle.remove()
            report['camera_weight_sum_min'] = min(checks.weight_sum_minima)
            report['camera_weight_sum_max'] = max(checks.weight_sum_maxima)
        else:
            runner = Runner.from_cfg(cfg)
            checks = SmokeChecks(args.baseline_checkpoint, report)
            runner.register_hook(checks)
            runner.train()
            # Exercise the post-warm-up stage without changing any weights.
            model = runner.model
            model.train()
            model.zero_grad(set_to_none=True)
            staged = next(h for h in runner.hooks
                          if h.__class__.__name__ == 'StagedFusionTrainingHook')
            staged._set_frozen(runner, False)
            batch = next(iter(runner.train_dataloader))
            data = model.data_preprocessor(batch, training=True)
            loss_dict = model(**data, mode='loss')
            loss, _ = model.parse_losses(loss_dict)
            assert torch.isfinite(loss)
            loss.backward()
            for group, prefixes in {
                'lidar_backbone': ('backbone.',),
                'detection_head': ('bbox_head.',),
            }.items():
                gradients = [p.grad for name, p in model.named_parameters()
                             if name.startswith(prefixes) and p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                assert any(g.abs().max() > 0 for g in gradients)
                report[f'unfrozen_{group}_gradient_passed'] = True
            model.zero_grad(set_to_none=True)
            report['checkpoint_written'] = (work_dir / 'epoch_1.pth').is_file()
            assert report['checkpoint_written']
        report['status'] = 'passed'
        write_json(work_dir / 'smoke_report.json', report)
        print('SMOKE PASSED:', args.mode, 'tokens=', args.tokens, flush=True)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        write_json(work_dir / 'smoke_report.json', report)
        raise


if __name__ == '__main__':
    main()
