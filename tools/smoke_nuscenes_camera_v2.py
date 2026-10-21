"""Bounded real-nuScenes GPU check of the larger V2 experiment, not AP.

Defaults to eight samples/iterations and two accumulated optimizer updates;
an optional longer bounded pilot checks initial optimization recovery. Then
checks joint detection-only backward, inference and strict checkpoint reload.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from smoke_camera_v2 import Checks


class NuScenesChecks(Checks):
    def before_train(self, runner):
        super().before_train(runner)
        model = runner.model
        batch = next(iter(runner.train_dataloader))
        processed = model.data_preprocessor(batch, training=True)
        model.eval()
        with torch.no_grad():
            actual = model.extract_feat(
                processed['inputs'], processed['data_samples'])
            expected = model.extract_pts_feat(
                processed['inputs']['voxels'], batch_size=1)
            for first, second in zip(actual, expected):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.report['initial_eval_matches_lidar_all_pyramid_levels'] = True
        model.train()
        staged = next(h for h in runner.hooks
                      if h.__class__.__name__ == 'StagedFusionTrainingHook')
        staged._set_frozen(runner, True)
        probability = model.fusion_module.lidar_spatial_drop_prob
        try:
            # Separate the production no-mask warm-up from the deliberately
            # harsh immediate-mask stress check. Zeroing BEV cells can produce
            # large but finite losses in a pretrained frozen detection head.
            with torch.no_grad():
                model.fusion_module.lidar_spatial_drop_prob = 0.0
                losses = model(**processed, mode='loss')
                warmup, _ = model.parse_losses(losses)
                model.fusion_module.lidar_spatial_drop_prob = probability
                losses = model(**processed, mode='loss')
                masked, _ = model.parse_losses(losses)
            assert torch.isfinite(warmup) and torch.isfinite(masked)
            self.report['initial_no_mask_warmup_loss'] = float(warmup)
            self.report['configured_mask_probability'] = probability
            self.report['initial_configured_mask_loss'] = float(masked)
            if probability > 0:
                self.report['initial_full_mask_stress_loss'] = float(masked)
        finally:
            model.fusion_module.lidar_spatial_drop_prob = probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--tokens', type=int, choices=(8, 16), default=8)
    parser.add_argument('--baseline-checkpoint')
    parser.add_argument('--no-mask', action='store_true',
                        help='Check the new no-hard-mask stability control.')
    parser.add_argument('--iterations', type=int, default=8,
                        help='8..512 iterations, divisible by accumulation 4.')
    args = parser.parse_args()
    if not 8 <= args.iterations <= 512 or args.iterations % 4:
        parser.error('--iterations must be 8..512 and divisible by four')
    work_dir = Path(args.work_dir).resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError('Use a new smoke work directory')
    work_dir.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', note='Execution check, not benchmark AP.')
    try:
        config = ('nuscenes_camera_v2_no_mask_smoke.py' if args.no_mask else
                  'nuscenes_camera_v2_smoke.py')
        cfg = Config.fromfile(Path(__file__).resolve().parents[1] /
                              'configs' / config)
        cfg.model.fusion_module.num_bottleneck_tokens = args.tokens
        if args.baseline_checkpoint:
            cfg.load_from = args.baseline_checkpoint
        if not Path(cfg.load_from).is_file():
            raise FileNotFoundError(cfg.load_from)
        cfg.work_dir, cfg.launcher, cfg.resume = str(work_dir), 'none', False
        cfg.train_cfg.max_iters = args.iterations
        cfg.default_hooks.checkpoint.interval = args.iterations
        cfg.train_dataloader.dataset.indices = args.iterations
        report.update(
            tokens=args.tokens, no_hard_mask=args.no_mask,
            physical_batch_size=1, accumulation_steps=4,
            requested_iterations=args.iterations,
            optimizer_updates=args.iterations // 4, gpu=torch.cuda.get_device_name(),
            baseline_checkpoint=str(Path(cfg.load_from).resolve()),
            baseline_sha256=hashlib.sha256(
                Path(cfg.load_from).read_bytes()).hexdigest())
        torch.cuda.reset_peak_memory_stats()
        runner = Runner.from_cfg(cfg)
        runner.register_hook(NuScenesChecks(
            report, cfg.load_from, expect_masking=not args.no_mask))
        runner.train()
        model = runner.model
        model.train()
        staged = next(h for h in runner.hooks
                      if h.__class__.__name__ == 'StagedFusionTrainingHook')
        staged._set_frozen(runner, False)
        model.zero_grad(set_to_none=True)
        batch = next(iter(runner.train_dataloader))
        processed = model.data_preprocessor(batch, training=True)
        losses = model(**processed, mode='loss')
        detection = {k: v for k, v in losses.items() if k != 'loss_camera_aux'}
        loss, _ = model.parse_losses(detection)
        assert torch.isfinite(loss)
        loss.backward()
        for name in ('backbone', 'bbox_head', 'img_backbone', 'img_neck'):
            grads = [p.grad for p in getattr(model, name).parameters()
                     if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), name
            assert any(g.abs().max() > 0 for g in grads), name
        assert model.fusion_module.bottleneck_tokens.grad.abs().max() > 0
        assert model.fusion_module.camera_aux_head.weight.grad is None
        report['detection_only_camera_and_lidar_backward_passed'] = True
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            predictions = model(**processed, mode='predict')
            features, output = model.extract_feat(
                processed['inputs'], processed['data_samples'],
                return_fusion_output=True)
        assert len(predictions) == 1
        pred = predictions[0].pred_instances_3d
        assert pred.scores_3d.numel() > 0, 'Warm-started detector returned no boxes'
        assert torch.isfinite(pred.scores_3d).all()
        assert torch.isfinite(pred.bboxes_3d.tensor).all()
        assert ((pred.labels_3d >= 0) & (pred.labels_3d < 10)).all()
        assert output['lidar_token_keep_mask'].all()
        assert all(torch.isfinite(f).all() for f in features)
        checkpoint = work_dir / f'iter_{args.iterations}.pth'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')[
            'state_dict'], strict=True)
        report.update(
            status='passed', prediction_passed=True, strict_reload_passed=True,
            eval_masking_disabled=True, checkpoint_written=checkpoint.is_file(),
            predicted_boxes=int(pred.scores_3d.numel()),
            parameters=sum(p.numel() for p in model.parameters()),
            peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (work_dir / 'smoke_report.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('NUScenes V2 SMOKE PASSED:', work_dir / 'smoke_report.json', flush=True)


if __name__ == '__main__':
    main()
