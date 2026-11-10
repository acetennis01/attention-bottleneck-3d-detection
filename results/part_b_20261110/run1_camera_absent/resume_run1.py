"""Audited epoch-boundary recovery of the existing camera-absent control."""
import fcntl
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules
from projects.myfusion.tools.smoke_transfusion_l_runner import equal_tree

BASE = Path('/home/yab/mmdetection3d')
RUN = BASE / 'work_dirs/part_b_kitti_absent_full_20261109'
ATTEMPT = Path(__file__).resolve().parent
CHECKPOINT = RUN / 'epoch_12.pth'
CONFIG = RUN / 'absent_resolved_config.py'
LAUNCHER = BASE / 'work_dirs/start_part_b_kitti_absent_full_20261109.py'
record = dict(status='preparing', checkpoint=str(CHECKPOINT),
              resumed_at=datetime.now().astimezone().isoformat(),
              restored_completed_epochs=12, restored_microbatches=14856,
              restored_optimizer_updates=1857,
              next_validation_epoch=14, epoch_12_validation_already_complete=True,
              random_state_note='Original seed and epoch sampler restored; bitwise dropout/point-shuffle RNG continuity is not claimed.')


def save():
    p = ATTEMPT / 'resume_report.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(p)


spec = importlib.util.spec_from_file_location('original_run1_launcher', LAUNCHER)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


class ResumeAudit(original.FrozenRunAudit):
    def before_train(self, runner):
        saved = torch.load(CHECKPOINT, map_location='cpu')
        assert runner.epoch == saved['meta']['epoch'] == 12
        assert runner.iter == saved['meta']['iter'] == 14856
        assert runner.iter % 8 == 0, 'No lost partial accumulation window permitted'
        equal_tree(runner.model.state_dict(), saved['state_dict'])
        equal_tree(runner.optim_wrapper.state_dict(), saved['optimizer'])
        equal_tree([s.state_dict() for s in runner.param_schedulers], saved['param_schedulers'])
        assert runner.optim_wrapper._accumulative_counts == 8
        self.steps = runner.iter // 8
        assert all(int(v['step']) == self.steps for v in saved['optimizer']['state'].values() if 'step' in v)
        baseline = torch.load(Config.fromfile(CONFIG).load_from, map_location='cpu')['state_dict']
        runner.model._remap_pointpillars_state_dict(baseline)
        self.fixed = {n: t.cpu().clone() for n, t in runner.model.state_dict().items()
                      if n.startswith(original.PREFIXES)}
        assert len(self.fixed) == 126
        for n, t in self.fixed.items():
            assert torch.equal(t, baseline[n]), n
        fusion = runner.model.fusion_module
        assert fusion.camera_mode == 'missing' and fusion.bypass_missing is False
        assert fusion.camera_drop_prob == .2 and fusion.use_bottleneck_gate
        self.absence_batches = 0
        self.feature_handle = fusion.register_forward_hook(self.check_absence)
        self.handle = runner.optim_wrapper.optimizer.register_step_post_hook(self.after_step)
        self.report = json.loads((ATTEMPT / 'before_resume_progress.json').read_text())
        self.report.update(status='training', completed_epochs=12, completed_microbatches=14856,
                           optimizer_updates=self.steps, resumed_from=str(CHECKPOINT),
                           resume_report=str(ATTEMPT / 'resume_report.json'),
                           checked_absence_batches_before_interruption=self.report.get('checked_absence_batches'),
                           checked_absence_batches=0, absence_counter_scope='current recovery process',
                           exact_model_optimizer_scheduler_restore=True)
        self.report.pop('error', None)
        record.update(status='restored', exact_model_optimizer_scheduler_restore=True,
                      protected_lidar_tensors_unchanged=len(self.fixed),
                      accumulation_boundary_verified=True,
                      optimizer_state_entries=len(saved['optimizer']['state']))
        self.write(runner)
        save()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        if batch_idx < 8 or (runner.iter + 1) % 100 == 0:
            record.update(status='training', completed_microbatches=runner.iter + 1,
                          optimizer_updates=self.steps, active_epoch=runner.epoch + 1,
                          checked_absence_batches=self.absence_batches)
            save()

    def after_train(self, runner):
        super().after_train(runner)
        record.update(status='complete', completed_microbatches=runner.iter,
                      optimizer_updates=self.steps,
                      completed_at=datetime.now().astimezone().isoformat())
        save()


if __name__ == '__main__':
    lock = (RUN.parent / 'part_b_kitti_absent_full_training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == '552caff03b7285137ef7fa0a48f5d2609bda7fde48c189ebe6f7ecfc3fb104f0'
        assert hashlib.sha256(LAUNCHER.read_bytes()).hexdigest() == '06f739d3b6622b2514edf2109fbf4cb482456a8ae063c6ad754ca0f6e03c259e'
        digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        assert digest == '576f5091ed95949890bc0e1ea111f7ed80523518961b8e207c90d7f3a5d6e66e'
        record['checkpoint_sha256'] = digest
        shutil.copy2(RUN / 'absent_training_progress.json', ATTEMPT / 'before_resume_progress.json')
        cfg = Config.fromfile(CONFIG)
        assert cfg.train_cfg.max_epochs == 40 and cfg.train_cfg.val_interval == 2
        assert cfg.custom_hooks[0].freeze_epochs == 40
        assert cfg.work_dir == str(RUN)
        assert cfg.train_dataloader.batch_size == 6 and cfg.optim_wrapper.accumulative_counts == 8
        cfg.resume = True
        cfg.load_from = str(CHECKPOINT)
        cfg.dump(ATTEMPT / 'resume_config.py')
        save()
        torch.set_num_threads(4)
        register_all_modules(init_default_scope=True)
        runner = Runner.from_cfg(cfg)
        audit = ResumeAudit()
        runner.register_hook(audit)
        runner.train()
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        save()
        if 'audit' in globals() and hasattr(audit, 'report'):
            audit.report.update(status='failed', error=repr(error))
            audit.write(runner)
        raise
