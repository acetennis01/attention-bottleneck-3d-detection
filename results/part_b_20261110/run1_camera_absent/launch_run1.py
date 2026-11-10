"""Full-budget camera-absent capacity control matched to completed gated KITTI."""
from pathlib import Path
from datetime import datetime
import fcntl,hashlib,json
import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules

BASE=Path('/home/yab/mmdetection3d')
RUN=BASE/'work_dirs/part_b_kitti_absent_full_20261109'
PREFLIGHT=BASE/'work_dirs/part_b_20261107_kitti_preflight/report.json'
PREFIXES=('voxel_encoder.','middle_encoder.','backbone.','neck.','bbox_head.')


class FrozenRunAudit(Hook):
    priority='LOWEST'
    def write(self,runner):
        p=Path(runner.work_dir)/'absent_training_progress.json'
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(self.report,indent=2)+'\n');tmp.replace(p)
    def before_train(self,runner):
        model=runner.model
        reference=torch.load(runner.cfg.load_from,map_location='cpu')['state_dict']
        model._remap_pointpillars_state_dict(reference)
        self.fixed={n:t.clone().cpu() for n,t in model.state_dict().items() if n.startswith(PREFIXES)}
        for n,t in self.fixed.items():assert torch.equal(t,reference[n]),n
        assert model.fusion_module.bottleneck_tokens.shape[1]==8
        assert model.fusion_module.use_bottleneck_gate
        assert all(not torch.count_nonzero(p) for gate in model.fusion_module.bottleneck_gates for p in gate.parameters())
        assert not runner.optim_wrapper.optimizer.state
        self.absence_batches=0
        self.feature_handle=model.fusion_module.register_forward_hook(self.check_absence)
        self.steps=0
        self.handle=runner.optim_wrapper.optimizer.register_step_post_hook(self.after_step)
        self.report=dict(status='training',started_at=datetime.now().astimezone().isoformat(),
            model='GatedCameraMBTBEVFusion',max_epochs=40,completed_epochs=0,
            planned_microbatches=runner.max_iters,accumulation=8,optimizer_updates=0,
            microbatches_per_epoch=len(runner.train_dataloader),protected_lidar_tensors=len(self.fixed),
            lidar_initializer_loaded_exactly=True,lidar_and_head_unchanged=True,
            baseline_sha256=hashlib.sha256(Path(runner.cfg.load_from).read_bytes()).hexdigest(),
            preflight_report=str(PREFLIGHT),fresh_optimizer=True,smoke_weights_used=False,
            control='camera-always-absent, trainable residual, no valid camera auxiliary targets')
        self.write(runner)
    def check_absence(self,module,inputs,output):
        assert not output['camera_available'].any()
        assert not output['image_valid_mask'].any()
        self.absence_batches+=1
    def after_step(self,*args):self.steps+=1
    def verify(self,runner):
        model=runner.model
        for name in ('voxel_encoder','middle_encoder','backbone','neck','bbox_head'):
            module=getattr(model,name)
            assert not any(m.training for m in module.modules()),name
            assert not any(p.requires_grad or p.grad is not None for p in module.parameters()),name
        for n,t in self.fixed.items():assert torch.equal(model.state_dict()[n].cpu(),t),n
    def before_train_epoch(self,runner):
        self.verify(runner)
        self.report['active_epoch']=runner.epoch+1;self.write(runner)
    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        assert torch.isfinite(torch.as_tensor(outputs['loss'])).all(),'Non-finite loss'
        if batch_idx<8 or (runner.iter+1)%100==0:
            self.report.update(completed_microbatches=runner.iter+1,optimizer_updates=self.steps,
                               latest_loss=float(outputs['loss'].detach()),
                               checked_absence_batches=self.absence_batches)
            self.write(runner)
    def after_train_epoch(self,runner):
        self.verify(runner)
        self.report.update(completed_epochs=runner.epoch+1,optimizer_updates=self.steps)
        self.write(runner)
    def after_train(self,runner):
        self.verify(runner)
        assert self.steps==runner.max_iters//8,(self.steps,runner.max_iters)
        self.report.update(status='complete',completed_epochs=runner.epoch,
            completed_microbatches=runner.max_iters,optimizer_updates=self.steps,
            lidar_and_head_unchanged=True,completed_at=datetime.now().astimezone().isoformat())
        self.handle.remove();self.feature_handle.remove();self.write(runner)


if __name__=='__main__':
    lock=(RUN.parent/'part_b_kitti_absent_full_training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight=json.loads(PREFLIGHT.read_text())
    assert preflight['status']=='passed' and preflight.get('architecture','gated')=='gated'
    assert preflight['strict_reload'] and preflight['real_data_missing_feature_identity']
    assert preflight['optimizer_updates']==4 and preflight['frozen_lidar_unchanged']
    if RUN.exists():raise FileExistsError('Use a new output directory; never overwrite an experiment')
    RUN.mkdir()
    cfg=Config.fromfile('projects/myfusion/configs/kitti_gated_absent_control.py')
    assert cfg.train_cfg.max_epochs==40 and cfg.custom_hooks[0].freeze_epochs==40
    # Keep the same RNG draw/probability as the full-camera arm, even though
    # camera_mode=missing makes all image information absent regardless.
    cfg.model.fusion_module.camera_drop_prob=.2
    assert cfg.model.fusion_module.camera_mode=='missing'
    assert cfg.model.fusion_module.bypass_missing is False
    assert cfg.model.fusion_module.type=='GatedCameraMBTBEVFusion'
    assert cfg.train_dataloader.batch_size==6 and cfg.optim_wrapper.accumulative_counts==8
    assert hashlib.sha256(Path(cfg.load_from).read_bytes()).hexdigest()=='296b97413d3f6813f6c75cb481a575ad10e9d1bee7126c8b2c7da41abd6d0d5b'
    cfg.work_dir=str(RUN);cfg.resume=False;cfg.launcher='none'
    cfg.val_evaluator.pklfile_prefix=str(RUN/'val_predictions')
    cfg.test_evaluator.pklfile_prefix=str(RUN/'test_predictions')
    reference_path=BASE/'work_dirs/part_b_kitti_gated_full_20261109/gated_resolved_config.py'
    reference=Config.fromfile(reference_path).to_dict()
    candidate=cfg.to_dict()
    for obj in (reference,candidate):
        for key in ('camera_mode','bypass_missing'):
            obj['model']['fusion_module'].pop(key)
        obj.pop('work_dir')
        obj['val_evaluator'].pop('pklfile_prefix')
        obj['test_evaluator'].pop('pklfile_prefix')
    assert candidate==reference,'Unexpected difference from completed gated KITTI recipe'
    (RUN/'matched_recipe_verification.json').write_text(json.dumps(dict(
        status='passed',reference_config=str(reference_path),
        reference_config_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        changed_fields=['camera always absent','residual kept trainable instead of bypassed','output paths'],
        max_epochs=40,seed=cfg.randomness.seed,physical_batch=6,accumulation=8,
        same_initializer=True,same_architecture=True,same_optimizer=True,same_scheduler=True,
        same_data=True,same_freeze=True,same_selection_rule=True,
        limitations='Absence removes camera auxiliary targets; retained residual receives detection gradients on all examples rather than bypassing dropped examples. This is an extra-capacity control, not perfect causal isolation.'),indent=2)+'\n')
    cfg.dump(RUN/'absent_resolved_config.py')
    torch.set_num_threads(4)
    register_all_modules(init_default_scope=True)
    runner=Runner.from_cfg(cfg)
    audit=FrozenRunAudit();runner.register_hook(audit)
    try:runner.train()
    except BaseException as error:
        if hasattr(audit,'report'):
            audit.report.update(status='failed',error=repr(error));audit.write(runner)
        raise
