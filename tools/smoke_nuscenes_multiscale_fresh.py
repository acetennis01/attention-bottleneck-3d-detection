"""Real-data cold initialization, training, inference and exact-resume smoke.

No benchmark AP and no production training are launched. Smoke outputs are
never used as initializers for the full run. Run from the MMDetection3D root.
"""

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.runner.checkpoint import CheckpointLoader

from smoke_camera_v2 import Checks


class FreshSmokeChecks(Checks):
    def before_train(self, runner):
        super().before_train(runner)
        model = runner.model
        def record_peak(gradient):
            self.report['fresh_training_observed_peak_allocated_mib'] = max(
                self.report.get('fresh_training_observed_peak_allocated_mib', 0),
                torch.cuda.max_memory_allocated() / 2**20)
        for parameter in model.parameters():
            if parameter.requires_grad:
                self.handles.append(parameter.register_hook(record_peak))
        imagenet = CheckpointLoader.load_checkpoint(
            'torchvision://resnet50', map_location='cpu')
        imagenet = imagenet.get('state_dict', imagenet)
        missing_counters = []
        for key, value in model.img_backbone.state_dict().items():
            if key not in imagenet and key.endswith('.num_batches_tracked'):
                assert torch.count_nonzero(value) == 0, key
                missing_counters.append(key)
                continue
            assert key in imagenet and torch.equal(value.cpu(), imagenet[key]), key
        self.report['exact_imagenet_camera_initialization'] = True
        self.report['imagenet_missing_batch_counters_initialized_zero'] = missing_counters
        batch = next(iter(runner.train_dataloader))
        self.processed = model.data_preprocessor(batch, training=True)
        model.eval()
        with torch.no_grad():
            actual = model.extract_feat(self.processed['inputs'], self.processed['data_samples'])
            expected = model.extract_pts_feat(self.processed['inputs']['voxels'], batch_size=1)
            for before, after in zip(expected, actual):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.report['all_initial_head_features_equal_lidar'] = True
        model.train()

    def after_train(self, runner):
        super().after_train(runner)
        model = runner.model
        model.eval()
        module = model.fusion_module
        for adapter in module.pyramid_residuals.values():
            assert adapter.weight.detach().abs().sum() > 0
        features = model.extract_feat(self.processed['inputs'], self.processed['data_samples'])
        for level in (0, 1, 2):
            scores = model.bbox_head.forward_single(features[level])[0]
            gradients = torch.autograd.grad(scores.square().mean(),
                (module.image_projection.weight, module.bottleneck_tokens), retain_graph=level < 2)
            assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
        self.report['detection_only_camera_and_bottleneck_gradients_all_three_scales'] = True
        with torch.no_grad():
            predictions = model(**self.processed, mode='predict')
        pred = predictions[0].pred_instances_3d
        assert pred.scores_3d.numel() and torch.isfinite(pred.scores_3d).all()
        assert torch.isfinite(pred.bboxes_3d.tensor).all()
        self.report['finite_detections'] = int(pred.scores_3d.numel())
        checkpoint = Path(runner.work_dir) / 'iter_8.pth'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
        self.report['strict_reload'] = True


class ResumeChecks(Hook):
    priority = 'LOWEST'

    def __init__(self, checkpoint, report):
        self.checkpoint, self.report = checkpoint, report

    def before_train(self, runner):
        saved = torch.load(self.checkpoint, map_location='cpu')
        assert runner.iter == 8
        for key, value in runner.model.state_dict().items():
            assert torch.equal(value.detach().cpu(), saved['state_dict'][key]), key
        optimizer = runner.optim_wrapper.optimizer.state_dict()
        reference = saved['optimizer']
        assert optimizer['param_groups'] == reference['param_groups']
        assert optimizer['state'].keys() == reference['state'].keys()
        for key in optimizer['state']:
            for field, value in optimizer['state'][key].items():
                previous = reference['state'][key][field]
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value.detach().cpu(), previous.cpu())
                else:
                    assert value == previous
        for actual, previous in zip(runner.param_schedulers, saved['param_schedulers']):
            assert actual.state_dict() == previous
        self.report['exact_model_optimizer_scheduler_resume_at_iter_8'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    args = parser.parse_args()
    output = Path(args.work_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new smoke directory')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', note='Execution and resume smoke, not benchmark AP')
    try:
        torch.set_num_threads(2)
        cfg = Config.fromfile(Path(__file__).resolve().parents[1] /
                              'configs/nuscenes_multiscale_fresh.py')
        cfg.work_dir, cfg.launcher = str(output), 'none'
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=8, val_interval=100000)
        cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
        cfg.test_cfg = cfg.test_dataloader = cfg.test_evaluator = None
        cfg.train_dataloader.dataset.indices = 16
        cfg.train_dataloader.num_workers = 0
        cfg.train_dataloader.persistent_workers = False
        # Keep the production warm-up/decay schedule; do not compress it.
        cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=8,
                                             by_epoch=False, save_last=True)
        cfg.default_hooks.logger.interval = 1
        cfg.custom_hooks = [hook for hook in cfg.custom_hooks
                            if hook.type != 'VersionedNuScenesValidationHook']
        runner = Runner.from_cfg(cfg)
        runner.register_hook(FreshSmokeChecks(report, cfg.load_from, expect_masking=False))
        runner.train()
        checkpoint = output / 'iter_8.pth'
        # Fresh runner must restore learned fusion and all optimizer/scheduler
        # state, while still verifying LiDAR against its original checkpoint.
        del runner
        cfg.resume = True
        cfg.load_from = str(checkpoint)
        cfg.train_cfg.max_iters = 16
        resumed = Runner.from_cfg(cfg)
        resumed.register_hook(ResumeChecks(checkpoint, report))
        resumed.train()
        integrity = json.loads((output / 'fresh_fusion_integrity.json').read_text())
        assert integrity['resumed'] and integrity['lidar_parameters_and_buffers_unchanged']
        assert integrity['optimizer_updates_this_session'] == 2
        assert integrity['total_optimizer_updates'] == 4
        report.update(status='passed', fresh_iterations=8, resumed_iterations=8,
                      total_optimizer_updates=4,
                      lidar_parameters_and_buffers_unchanged=True,
                      smoke_checkpoint_used_for_full_training=False)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (output / 'smoke_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('FRESH MULTISCALE SMOKE PASSED', output, flush=True)


if __name__ == '__main__':
    main()
