"""Eight real batches from the selected MBT model; no benchmark validation.

Unlike cold-start smokes, the trained residual must NOT be assumed zero.
This verifies an exact full-model warm start and permanently fixed LiDAR.
Smoke checkpoint/optimizer state must never initialize the actual pilot.
"""

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner


class SmokeChecks(Hook):
    priority = 'NORMAL'

    def __init__(self, report, checkpoint):
        self.report = report
        self.checkpoint = checkpoint
        self.handles = []
        self.gradients = {}
        self.initial = {}

    def before_train(self, runner):
        model = runner.model
        saved = torch.load(self.checkpoint, map_location='cpu')['state_dict']
        actual = model.state_dict()
        assert set(saved) == set(actual)
        for key, value in actual.items():
            assert torch.equal(value.detach().cpu(), saved[key]), key
        self.report['exact_full_mbt_warm_start'] = True
        self.report['losses'] = []
        for name in ('img_backbone', 'img_neck', 'fusion_module'):
            self.initial[name] = {k: v.detach().cpu().clone() for k, v in
                                  getattr(model, name).named_parameters()}

        def capture(name):
            def check(gradient):
                assert torch.isfinite(gradient).all(), name
                if gradient.abs().max() > 0:
                    self.gradients[name] = True
            return check

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                group = ('bottleneck' if name == 'fusion_module.bottleneck_tokens'
                         else 'camera_aux' if name.startswith('fusion_module.camera_aux_head.')
                         else name.split('.')[0])
                self.handles.append(parameter.register_hook(capture(group)))
        self.handles.append(model.fusion_module.register_forward_hook(self.check_features))

    @staticmethod
    def check_features(module, _inputs, output):
        assert module.lidar_spatial_drop_prob == 0
        assert output['lidar_token_keep_mask'].all()
        assert torch.isfinite(output['bev_features']).all()
        assert .999 <= float(output['camera_weight_sum_min']) <= 1.001
        assert .999 <= float(output['camera_weight_sum_max']) <= 1.001

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        loss = float(outputs['loss'].detach())
        assert torch.isfinite(torch.tensor(loss))
        self.report['losses'].append(loss)

    def after_train(self, runner):
        for name, previous in self.initial.items():
            current = dict(getattr(runner.model, name).named_parameters())
            assert any(not torch.equal(value, current[key].detach().cpu())
                       for key, value in previous.items()), name
        for name in ('img_backbone', 'img_neck', 'fusion_module', 'bottleneck', 'camera_aux'):
            assert self.gradients.get(name), name
        for name in ('voxel_encoder', 'backbone', 'neck', 'bbox_head'):
            assert not self.gradients.get(name), name
        self.report['camera_fpn_fusion_aux_updates'] = True
        self.report['nonzero_finite_gradients'] = self.gradients
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    args = parser.parse_args()
    output = Path(args.work_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty smoke work directory')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', note='Execution test, not benchmark AP.')
    try:
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.35)
        cfg = Config.fromfile(Path(__file__).resolve().parents[1] /
                              'configs/nuscenes_camera_v2_frozen_lidar_pilot.py')
        cfg.work_dir, cfg.launcher = str(output), 'none'
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=8, val_interval=10000)
        cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
        cfg.test_cfg = cfg.test_dataloader = cfg.test_evaluator = None
        cfg.param_scheduler = []
        cfg.train_dataloader.dataset.indices = 8
        cfg.train_dataloader.num_workers = 0
        cfg.train_dataloader.persistent_workers = False
        cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=8,
                                             by_epoch=False, save_last=True)
        cfg.default_hooks.logger.interval = 1
        runner = Runner.from_cfg(cfg)
        checks = SmokeChecks(report, cfg.load_from)
        runner.register_hook(checks)
        runner.train()
        guard = next(h for h in runner.hooks if h.__class__.__name__ == 'FrozenLiDARIntegrityHook')
        guard._check_state(runner)
        guard._check_modes_and_gradients(runner)
        report['all_lidar_head_parameters_and_bn_buffers_unchanged'] = True

        model = runner.model
        model.zero_grad(set_to_none=True)
        model.train()
        staged = next(h for h in runner.hooks if h.__class__.__name__ == 'StagedFusionTrainingHook')
        staged._set_frozen(runner, True)
        processed = model.data_preprocessor(next(iter(runner.train_dataloader)), training=True)
        losses = model(**processed, mode='loss')
        detection = {key: value for key, value in losses.items() if key != 'loss_camera_aux'}
        loss, _ = model.parse_losses(detection)
        assert torch.isfinite(loss)
        loss.backward()
        for name in ('img_backbone', 'img_neck'):
            gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(g.abs().max() > 0 for g in gradients), name
        assert model.fusion_module.bottleneck_tokens.grad.abs().max() > 0
        assert model.fusion_module.camera_aux_head.weight.grad is None
        guard._check_modes_and_gradients(runner)
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode():
            predictions = model(**processed, mode='predict')
        pred = predictions[0].pred_instances_3d
        assert pred.scores_3d.numel() and torch.isfinite(pred.scores_3d).all()
        assert torch.isfinite(pred.bboxes_3d.tensor).all()
        assert ((pred.labels_3d >= 0) & (pred.labels_3d < 10)).all()
        model.load_state_dict(torch.load(output / 'iter_8.pth', map_location='cpu')['state_dict'], strict=True)
        guard._check_state(runner)
        report.update(status='passed', detection_only_camera_and_bottleneck_backward=True,
                      strict_reload=True, predicted_boxes=int(pred.scores_3d.numel()),
                      peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      iterations=8, optimizer_updates=2, hard_mask_disabled=True)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (output / 'smoke_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('FROZEN CONTINUATION SMOKE PASSED', output, flush=True)


if __name__ == '__main__':
    main()
