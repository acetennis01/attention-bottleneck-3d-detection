"""Bounded, real-KITTI V2 GPU check; does not produce benchmark AP.

Run from the MMDetection3D root. All outputs go into a new, explicitly supplied
work directory; existing dataset metadata and training runs are not modified.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.config import Config
from mmengine.runner import Runner


LIDAR_PREFIXES = ('voxel_encoder.', 'middle_encoder.', 'backbone.',
                  'neck.', 'bbox_head.')


def component(name):
    if name.startswith('img_backbone.'):
        return 'image_backbone'
    if name.startswith('img_neck.'):
        return 'image_neck'
    if name.startswith('fusion_module.bottleneck_tokens'):
        return 'bottleneck'
    if name.startswith('fusion_module.camera_aux_head.'):
        return 'camera_aux'
    if name.startswith('fusion_module.bev_residual.'):
        return 'bev_residual'
    if name.startswith(LIDAR_PREFIXES):
        return 'lidar_detector'
    return 'fusion_other'


class Checks(Hook):
    priority = 'LOW'

    def __init__(self, report, baseline, expect_masking=True):
        self.report, self.baseline = report, baseline
        self.expect_masking = bool(expect_masking)
        self.handles, self.losses, self.keep_ratios = [], [], []
        self.grad_max, self.valid_ratios = {}, []

    def before_train(self, runner):
        model = runner.model
        source = torch.load(self.baseline, map_location='cpu')['state_dict']
        model._remap_pointpillars_state_dict(source)
        actual = model.state_dict()
        self.frozen = {
            name: value.detach().cpu().clone()
            for name, value in actual.items() if name.startswith(LIDAR_PREFIXES)}
        assert self.frozen, 'No LiDAR parameters found'
        for name, value in self.frozen.items():
            assert name in source and torch.equal(value, source[name]), name
        self.start = {
            name: value.detach().cpu().clone()
            for name, value in model.named_parameters() if value.requires_grad}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                group = component(name)

                def capture(gradient, group=group):
                    assert torch.isfinite(gradient).all(), group
                    self.grad_max[group] = max(
                        self.grad_max.get(group, 0.0),
                        gradient.detach().abs().max().item())

                self.handles.append(parameter.register_hook(capture))
        self.handles.append(
            model.fusion_module.register_forward_hook(self.capture_features))

    def capture_features(self, module, inputs, output):
        assert torch.isfinite(output['bev_features']).all()
        assert 0.999 <= float(output['camera_weight_sum_min']) <= 1.001
        assert 0.999 <= float(output['camera_weight_sum_max']) <= 1.001
        self.valid_ratios.append(float(output['image_valid_ratio']))
        self.keep_ratios.append(float(output['lidar_token_keep_ratio']))

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        value = float(outputs['loss'].detach())
        assert torch.isfinite(torch.tensor(value))
        self.losses.append(value)

    def after_train(self, runner):
        model = runner.model
        for name, value in self.frozen.items():
            assert torch.equal(model.state_dict()[name].cpu(), value), name
        changed = {}
        for name, parameter in model.named_parameters():
            if name in self.start and not torch.equal(
                    self.start[name], parameter.detach().cpu()):
                group = component(name)
                changed[group] = changed.get(group, 0) + 1
        for group in ('image_backbone', 'image_neck', 'bottleneck',
                      'camera_aux', 'bev_residual'):
            assert self.grad_max.get(group, 0) > 0, f'No {group} gradient'
            assert changed.get(group, 0) > 0, f'No {group} update'
        assert min(self.valid_ratios) > 0
        if self.expect_masking:
            assert min(self.keep_ratios) < 1, 'Coupled masking was not exercised'
        else:
            assert all(value == 1.0 for value in self.keep_ratios), (
                'No-mask experiment unexpectedly masked LiDAR tokens')
        self.report.update(
            lidar_checkpoint_loaded_exactly=True, frozen_lidar_unchanged=True,
            train_iterations=len(self.losses), losses=self.losses,
            gradient_max_abs=self.grad_max,
            changed_parameter_tensors=changed,
            visible_camera_ratio_min=min(self.valid_ratios),
            expected_masking=self.expect_masking,
            lidar_keep_ratio_min=min(self.keep_ratios))
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--baseline-checkpoint')
    parser.add_argument('--tokens', type=int,
                        help='Override only bottleneck count for a V2 smoke.')
    args = parser.parse_args()
    if args.tokens is not None and args.tokens < 1:
        parser.error('--tokens must be positive')
    work_dir = Path(args.work_dir).resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError('Use a new smoke work directory')
    work_dir.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', note='Execution smoke, not benchmark AP.')
    try:
        cfg = Config.fromfile(
            Path(__file__).resolve().parents[1] /
            'configs/kitti_camera_v2_smoke.py')
        if args.baseline_checkpoint:
            cfg.load_from = args.baseline_checkpoint
        if args.tokens is not None:
            cfg.model.fusion_module.num_bottleneck_tokens = args.tokens
        if not Path(cfg.load_from).is_file():
            raise FileNotFoundError(cfg.load_from)
        cfg.work_dir = str(work_dir)
        cfg.launcher, cfg.resume = 'none', False
        # Limit dataset construction as well as iterations. The official
        # annotations are read, never rewritten for a smoke subset.
        dataset = cfg.train_dataloader.dataset
        while 'dataset' in dataset:
            dataset = dataset.dataset
        dataset.indices = cfg.train_dataloader.batch_size * cfg.train_cfg.max_iters
        report.update(
            bottleneck_tokens=cfg.model.fusion_module.num_bottleneck_tokens,
            physical_batch_size=cfg.train_dataloader.batch_size,
            accumulation_steps=cfg.optim_wrapper.accumulative_counts,
            optimizer_updates=(cfg.train_cfg.max_iters //
                               cfg.optim_wrapper.accumulative_counts),
            gpu=torch.cuda.get_device_name(), pytorch=torch.__version__,
            baseline_checkpoint=str(Path(cfg.load_from).resolve()),
            baseline_sha256=hashlib.sha256(
                Path(cfg.load_from).read_bytes()).hexdigest())
        torch.cuda.reset_peak_memory_stats()
        runner = Runner.from_cfg(cfg)
        checks = Checks(report, cfg.load_from)
        runner.register_hook(checks)
        runner.train()
        model = runner.model
        report['parameters'] = sum(p.numel() for p in model.parameters())
        # Verify an actual post-warm-up backward pass without another update.
        model.train()
        staged = next(hook for hook in runner.hooks
                      if hook.__class__.__name__ == 'StagedFusionTrainingHook')
        staged._set_frozen(runner, False)
        model.zero_grad(set_to_none=True)
        batch = next(iter(runner.train_dataloader))
        processed = model.data_preprocessor(batch, training=True)
        losses = model(**processed, mode='loss')
        # Exclude the auxiliary task: these camera gradients must come from
        # the actual detection objective through the MBT path.
        detection_losses = {name: value for name, value in losses.items()
                            if name != 'loss_camera_aux'}
        loss, _ = model.parse_losses(detection_losses)
        loss.backward()
        for name in ('backbone', 'bbox_head', 'img_backbone', 'img_neck'):
            gradients = [p.grad for p in getattr(model, name).parameters()
                         if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(g.abs().max() > 0 for g in gradients)
        assert model.fusion_module.bottleneck_tokens.grad.abs().max() > 0
        report['unfrozen_lidar_backward_passed'] = True
        report['detection_only_camera_backward_passed'] = True
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            predictions = model(**processed, mode='predict')
        assert len(predictions) == cfg.train_dataloader.batch_size
        for sample in predictions:
            assert torch.isfinite(sample.pred_instances_3d.scores_3d).all()
            assert torch.isfinite(sample.pred_instances_3d.bboxes_3d.tensor).all()
        report['prediction_passed'] = True
        report['checkpoint_written'] = (work_dir / 'iter_16.pth').is_file()
        assert report['checkpoint_written']
        report['peak_gpu_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (work_dir / 'smoke_report.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('V2 SMOKE PASSED:', work_dir / 'smoke_report.json', flush=True)


if __name__ == '__main__':
    main()
