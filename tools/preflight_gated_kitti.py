"""Bounded real-data check of frozen KITTI gated/full attention; not AP."""
import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules
from projects.myfusion.tools.smoke_camera_v2 import Checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--architecture', choices=('gated', 'full_attention'),
                        default='gated')
    args = parser.parse_args()
    root = Path(args.work_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new preflight directory')
    root.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', smoke_only=True)
    try:
        torch.set_num_threads(4)
        register_all_modules(init_default_scope=True)
        filename = ('kitti_gated_b8.py' if args.architecture == 'gated'
                    else 'kitti_full_attention.py')
        cfg = Config.fromfile('projects/myfusion/configs/' + filename)
        cfg.work_dir = str(root)
        cfg.launcher = 'none'
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=32,
                             val_interval=1000)
        cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
        cfg.test_cfg = cfg.test_dataloader = cfg.test_evaluator = None
        cfg.param_scheduler = []
        cfg.train_dataloader.num_workers = 0
        cfg.train_dataloader.persistent_workers = False
        dataset = cfg.train_dataloader.dataset
        while 'dataset' in dataset:
            dataset = dataset.dataset
        dataset.indices = 192
        cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=32,
            by_epoch=False, save_last=True, max_keep_ckpts=1, save_best=None)
        runner = Runner.from_cfg(cfg)
        groups = None if args.architecture == 'gated' else (
            'image_backbone', 'image_neck', 'camera_aux', 'bev_residual',
            'fusion_other')
        runner.register_hook(Checks(report, cfg.load_from, expect_masking=False,
                                    required_gradient_groups=groups))
        runner.train()
        model = runner.model
        saved = torch.load(root / 'iter_32.pth', map_location='cpu')
        model.load_state_dict(saved['state_dict'], strict=True)
        model.eval()
        model.zero_grad(set_to_none=True)
        batch = next(iter(runner.train_dataloader))
        processed = model.data_preprocessor(batch, training=True)
        losses = model(**processed, mode='loss')
        loss, _ = model.parse_losses({k: v for k, v in losses.items()
                                     if k != 'loss_camera_aux'})
        loss.backward()
        fusion = model.fusion_module
        gradient_groups = dict(camera=model.img_backbone.parameters())
        if args.architecture == 'gated':
            gradient_groups.update(bottleneck=[fusion.bottleneck_tokens],
                                   gates=fusion.bottleneck_gates.parameters())
        else:
            assert not any('bottleneck' in name for name in fusion.state_dict())
            gradient_groups.update(image_attention=fusion.image_blocks.parameters(),
                                   lidar_attention=fusion.lidar_blocks.parameters(),
                                   readout=fusion.lidar_readout.parameters())
        for name, parameters in gradient_groups.items():
            grads = [p.grad for p in parameters if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), name
            assert any(g.abs().max() > 0 for g in grads), name
        model.zero_grad(set_to_none=True)
        captures = []

        def identity_check(module, inputs, output):
            assert torch.equal(inputs[1], output['bev_features'])
            captures.append(True)

        model.fusion_module.camera_mode = 'missing'
        handle = model.fusion_module.register_forward_hook(identity_check)
        with torch.no_grad():
            predictions = model(**processed, mode='predict')
        handle.remove()
        assert captures
        for sample in predictions:
            assert torch.isfinite(sample.pred_instances_3d.scores_3d).all()
            assert torch.isfinite(sample.pred_instances_3d.bboxes_3d.tensor).all()
        report.update(status='passed', strict_reload=True,
            architecture=args.architecture,
            detection_only_gradient_groups=list(gradient_groups),
            real_data_missing_feature_identity=True,
            missing_prediction_passed=True, microbatches=32, optimizer_updates=4)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
