"""Native MMEngine cold-start/save/resume smoke; never seeds a full run."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules


def equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual.detach().cpu(), expected.cpu())
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            equal_tree(first, second)
    else:
        assert actual == expected


class ResumeChecks(Hook):
    priority = 'LOWEST'

    def __init__(self, checkpoint, report):
        self.checkpoint = checkpoint
        self.report = report

    def before_train(self, runner):
        saved = torch.load(self.checkpoint, map_location='cpu')
        assert runner.iter == 8
        equal_tree(runner.model.state_dict(), saved['state_dict'])
        # MMEngine persists base_param_settings alongside the AdamW state.
        # Compare the complete wrapper, not just torch.optim's two-key dict.
        equal_tree(runner.optim_wrapper.state_dict(), saved['optimizer'])
        equal_tree([item.state_dict() for item in runner.param_schedulers],
                   saved['param_schedulers'])
        self.report['exact_model_optimizer_scheduler_restore_at_iter_8'] = True
        self.report['restored_optimizer_state_entries'] = len(saved['optimizer']['state'])


def smoke_config(base, work_dir, maximum):
    cfg = deepcopy(base)
    cfg.work_dir = str(work_dir)
    cfg.launcher = 'none'
    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=maximum,
                         val_begin=8, val_interval=8)
    # Keep production LR schedule, optimizer, accumulation and all audit hooks.
    cfg.train_dataloader.dataset.indices = 16
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.val_dataloader.dataset.indices = 2
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    cfg.val_evaluator.format_only = True
    cfg.test_cfg = cfg.test_dataloader = cfg.test_evaluator = None
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=8,
        by_epoch=False, save_last=True, max_keep_ckpts=2, save_best=None)
    cfg.default_hooks.logger.interval = 1
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--config', default=(
        'projects/myfusion/configs/transfusion_l_mbt_swint_nuscenes.py'))
    args = parser.parse_args()
    root = Path(args.work_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a fresh smoke output directory')
    root.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', smoke_only=True,
        note='16 microbatches across two runner instances, 4 optimizer updates. '
             'Two-frame format-only validation is not benchmark evaluation. '
             'No smoke checkpoint may initialize the full run.')
    try:
        torch.set_num_threads(4)
        register_all_modules(init_default_scope=True)
        base = Config.fromfile(args.config)
        first_cfg = smoke_config(base, root / 'cold', 8)
        first = Runner.from_cfg(first_cfg)
        first.train()
        cold_progress = json.loads((root / 'cold/training_progress.json').read_text())
        assert cold_progress['status'] == 'complete'
        assert cold_progress['optimizer_updates_this_session'] == 2
        assert cold_progress['exact_cold_camera_initialization']
        checkpoint = root / 'cold/iter_8.pth'
        saved = torch.load(checkpoint, map_location='cpu')
        assert saved['meta']['iter'] == 8 and saved['optimizer']['state']
        assert saved['param_schedulers']
        first.model.load_state_dict(saved['state_dict'], strict=True)
        report.update(cold_start=True, cold_actual_optimizer_steps=2,
                      strict_native_checkpoint_reload=True,
                      cold_progress=cold_progress)
        del first, saved
        gc.collect()
        torch.cuda.empty_cache()
        second_cfg = smoke_config(base, root / 'resumed', 16)
        second_cfg.resume = True
        second_cfg.load_from = str(checkpoint)
        second = Runner.from_cfg(second_cfg)
        second.register_hook(ResumeChecks(checkpoint, report))
        second.train()
        progress = json.loads((root / 'resumed/training_progress.json').read_text())
        assert progress['resumed'] and progress['status'] == 'complete'
        assert progress['start_microbatch'] == 8
        assert progress['optimizer_updates_this_session'] == 2
        assert progress['total_optimizer_updates'] == 4
        for path in (root / 'cold/validation/iter_000008/metrics.json',
                     root / 'resumed/validation/iter_000016/metrics.json'):
            assert json.loads(path.read_text())['frames'] == 2
        for directory in ('cold', 'resumed'):
            integrity = json.loads((root / directory / 'transfusion_l_integrity.json').read_text())
            assert integrity['lidar_and_head_unchanged']
        report.update(status='passed', resumed_progress=progress,
            total_microbatches=16, total_optimizer_updates=4,
            resumed_actual_optimizer_steps=2,
            two_frame_format_only_validation_both_phases=True,
            lidar_and_head_parameters_and_bn_buffers_unchanged=True,
            future_augmentation_rng_bitwise_reproduction_claimed=False)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
