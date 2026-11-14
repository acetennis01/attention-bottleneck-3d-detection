"""Matched two-dataset gated studies with isolated configs and resumable jobs.

Run from the MMDetection3D root. No historical run is modified.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from projects.myfusion.tools.gated_study_plan import study_arms, apply_arm, changed_paths, cold_start_reference, aligned_kitti_checkpoints

ROOT=Path('/home/yab/mmdetection3d')
REFERENCES={
 'kitti': ROOT/'work_dirs/part_b_kitti_gated_full_20261109/gated_resolved_config.py',
 'nuscenes': ROOT/'work_dirs/part_b_20261107_transfusion_gated_full/transfusion_gated_full.py'}
LIDAR_SHA='296b97413d3f6813f6c75cb481a575ad10e9d1bee7126c8b2c7da41abd6d0d5b'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(path)

def plan(root):
    from mmengine.config import Config
    entries=[]
    for dataset in REFERENCES:
        saved=Config.fromfile(REFERENCES[dataset])
        if dataset=='nuscenes':
            canonical=Config.fromfile(Path(__file__).resolve().parents[1]/'configs/transfusion_l_gated_swint_nuscenes.py')
            allowed_resume={'resume','load_from','launcher','work_dir',
                'val_evaluator.jsonfile_prefix','test_evaluator.jsonfile_prefix'}
            assert set(changed_paths(canonical.to_dict(),saved.to_dict()))<=allowed_resume
        base=Config(cold_start_reference(saved.to_dict(),dataset))
        assert base.model.fusion_module.num_bottleneck_tokens==8
        assert base.model.fusion_module.embed_dim==256
        assert base.model.fusion_module.use_bottleneck_gate
        assert base.randomness.seed==0
        if dataset=='kitti':
            assert base.train_cfg.max_epochs==40 and base.train_dataloader.batch_size==6
            assert base.optim_wrapper.accumulative_counts==8
            assert sha(base.load_from)==LIDAR_SHA
            assert base.custom_hooks[0].freeze_epochs==40
        else:
            assert base.train_cfg.max_iters==160000 and base.train_dataloader.batch_size==1
            assert base.optim_wrapper.accumulative_counts==4 and base.model.freeze_lidar
            assert base.model.fusion_module.camera_view_drop_prob==.15
        for arm in study_arms(dataset):
            name=dataset+'_'+arm['name'];dest=root/'runs'/name
            cfg=Config(apply_arm(base.to_dict(),arm))
            cfg.work_dir=str(dest);cfg.resume=False;cfg.launcher='none'
            if dataset=='kitti':
                # 1238 microbatches/epoch; four epochs align with accumulation=8.
                # Best-score saves remain driven by validation every two epochs.
                cfg.default_hooks.checkpoint.interval=4
            for field,prefix in [('val_evaluator','validation'),('test_evaluator','test')]:
                if dataset=='kitti':cfg[field].pklfile_prefix=str(dest/(prefix+'_predictions'))
                else:cfg[field].jsonfile_prefix=str(dest/prefix)
            # Output paths, explicitly reset resume state, and intervention fields only.
            diffs=changed_paths(base.to_dict(),cfg.to_dict())
            allowed={'default_hooks.checkpoint.interval','work_dir','resume','launcher','val_evaluator.pklfile_prefix',
                'test_evaluator.pklfile_prefix','val_evaluator.jsonfile_prefix','test_evaluator.jsonfile_prefix'}
            allowed.update('model.fusion_module.'+x for x in arm['changes'])
            assert set(diffs)<=allowed,(name,diffs)
            path=root/'configs'/(name+'.py');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():assert Config.fromfile(path).to_dict()==cfg.to_dict(),name
            else:cfg.dump(path)
            entries.append(dict(name=name,dataset=dataset,arm=arm['name'],changes=arm['changes'],
                config=str(path),config_sha256=sha(path),work_dir=str(dest),
                reference_config=str(REFERENCES[dataset]),reference_config_sha256=sha(REFERENCES[dataset]),
                recipe_differences=diffs,status='pending'))
    return entries

def run_arm(args):
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.hooks import Hook
    from mmengine.dataset import pseudo_collate
    from mmdet3d.utils import register_all_modules
    from projects.myfusion.tools.smoke_camera_v2 import Checks
    torch.set_num_threads(4);register_all_modules(init_default_scope=True)
    cfg=Config.fromfile(args.config);dataset=args.dataset
    dest=Path(args.output) if args.output else Path(cfg.work_dir)
    smoke=args.mode=='preflight'
    if smoke:
        assert not dest.exists(),dest;dest.mkdir(parents=True)
        cfg.work_dir=str(dest);cfg.resume=False
        cfg.train_cfg=dict(type='IterBasedTrainLoop',max_iters=32 if dataset=='kitti' else 16,val_interval=1000000)
        if dataset=='kitti':
            cfg.val_cfg=cfg.val_dataloader=cfg.val_evaluator=None
        else:
            # Initializer audit reads the original ann_file even without validation.
            cfg.val_dataloader.dataset.indices=2
            cfg.val_dataloader.num_workers=0;cfg.val_dataloader.persistent_workers=False
            cfg.val_evaluator.format_only=True
        cfg.test_cfg=cfg.test_dataloader=cfg.test_evaluator=None
        # Preserve the production optimizer/scheduler shape for a native resume check.
        cfg.train_dataloader.num_workers=0;cfg.train_dataloader.persistent_workers=False
        ds=cfg.train_dataloader.dataset
        while 'dataset' in ds:ds=ds.dataset
        ds.indices=192 if dataset=='kitti' else 16
        cfg.default_hooks.checkpoint=dict(type='CheckpointHook',by_epoch=False,
            interval=cfg.train_cfg.max_iters,save_last=True,max_keep_ckpts=1,save_best=None)
        cfg.default_hooks.logger.interval=8
    else:
        pre=json.loads(Path(args.preflight).read_text());assert pre['status']=='passed'
        assert pre['production_config_sha256']==sha(args.config)
        if dest.exists():
            progress=dest/'study_progress.json'
            if progress.exists() and json.loads(progress.read_text()).get('status')=='complete':return
            if dataset=='kitti':
                candidates=aligned_kitti_checkpoints(dest)
                assert candidates, 'No accumulation-aligned full checkpoint; preserve and inspect the interrupted attempt'
                cp=candidates[-1]
            else:
                last=dest/'last_checkpoint'
                assert last.exists(),'Existing run has no durable checkpoint; inspect rather than restart'
                cp=Path(last.read_text().strip());cp=cp if cp.is_absolute() else ROOT/cp
            assert cp.is_file() and cp.resolve().is_relative_to(dest.resolve())
            cfg.resume=True;cfg.load_from=str(cp)
        else:dest.mkdir(parents=True)
    report=dict(status='running',smoke_only=smoke,production_config_sha256=sha(args.config))
    class Audit(Hook):
        priority='LOWEST'
        def before_train(self,r):
            self.start=int(r.iter);self.steps=0;self.acc=cfg.optim_wrapper.accumulative_counts
            if not smoke:assert r.max_iters==(49520 if dataset=='kitti' else 160000)
            assert self.start%self.acc==0
            m=r.model
            self.names=('voxel_encoder','middle_encoder','backbone','neck','bbox_head') if dataset=='kitti' else tuple(m.lidar_module_names)
            if dataset=='kitti':
                for name in self.names:
                    getattr(m,name).requires_grad_(False);getattr(m,name).eval()
            prefixes=tuple(x+'.' for x in self.names)
            self.fixed={n:t.detach().cpu().clone() for n,t in m.state_dict().items() if n.startswith(prefixes)}
            if dataset=='kitti':
                base_cfg=Config.fromfile(REFERENCES['kitti']);raw=torch.load(base_cfg.load_from,map_location='cpu')['state_dict'];m._remap_pointpillars_state_dict(raw)
                for n,t in self.fixed.items():assert torch.equal(t,raw[n]),n
            if self.start==0:
                assert not r.optim_wrapper.optimizer.state
                assert not any(torch.count_nonzero(p) for p in m.fusion_module.bev_residual.parameters())
                for gate in m.fusion_module.bottleneck_gates:
                    assert not any(torch.count_nonzero(p) for p in gate.parameters())
            self.handle=r.optim_wrapper.optimizer.register_step_post_hook(lambda *a:setattr(self,'steps',self.steps+1))
            report.update(status='training',start_microbatch=self.start,protected_tensors=len(self.fixed),
                started_at=datetime.now(timezone.utc).isoformat(),resumed=bool(cfg.resume),
                expected_total_updates=(6190 if dataset=='kitti' and not smoke else
                    (cfg.train_cfg.max_iters if smoke or dataset=='nuscenes' else 49520)//self.acc))
            self.verify(r);write(dest/'study_progress.json',report)
        def verify(self,r):
            for name in self.names:
                m=getattr(r.model,name)
                assert not any(p.requires_grad for p in m.parameters()),name
                assert not any(x.training for x in m.modules()),name
            for n,t in self.fixed.items():assert torch.equal(r.model.state_dict()[n].detach().cpu(),t),n
        def after_train_iter(self,r,batch_idx,data_batch=None,outputs=None):
            assert torch.isfinite(torch.as_tensor(outputs['loss'])).all()
            if (r.iter+1)%100==0 or r.iter<self.start+8:
                report.update(completed_microbatches=r.iter+1,optimizer_updates=self.start//self.acc+self.steps,
                    latest_loss=float(torch.as_tensor(outputs['loss']).detach()),updated_at=datetime.now(timezone.utc).isoformat())
                write(dest/'study_progress.json',report)
        def after_train_epoch(self,r):
            self.verify(r);report['completed_epochs']=r.epoch+1
            if not smoke and dataset=='kitti' and (r.epoch+1)%4==0:
                # Save before validation can update last_checkpoint after a best score.
                native=dest/f'epoch_{r.epoch+1}.pth';archive=dest/f'aligned_epoch_{r.epoch+1}.pth'
                assert native.exists(), native
                if not archive.exists():
                    tmp=archive.with_suffix('.pth.partial');shutil.copy2(native,tmp);tmp.replace(archive)
                report['aligned_recovery_checkpoint']=str(archive)
            write(dest/'study_progress.json',report)
        def after_train(self,r):
            self.verify(r);expected=r.max_iters//self.acc
            assert self.start//self.acc+self.steps==expected
            report.update(status='complete',completed_microbatches=r.max_iters,optimizer_updates=expected,
                frozen_lidar_and_head_unchanged=True,completed_at=datetime.now(timezone.utc).isoformat())
            write(dest/'study_progress.json',report);self.handle.remove()
    runner=Runner.from_cfg(cfg);runner.register_hook(Audit())
    if smoke:runner.register_hook(Checks(report,Config.fromfile(REFERENCES['kitti']).load_from,expect_masking=False)) if dataset=='kitti' else None
    try:
        runner.train()
        if smoke:
            cp=dest/f'iter_{cfg.train_cfg.max_iters}.pth';saved=torch.load(cp,map_location='cpu')
            runner.model.load_state_dict(saved['state_dict'],strict=True)
            # Native checkpoint persists optimizer and scheduler state for recovery.
            assert saved['optimizer']['state'] and saved['param_schedulers']
            runner.model.eval();runner.model.zero_grad(set_to_none=True)
            batch=next(iter(runner.train_dataloader));processed=runner.model.data_preprocessor(batch,training=False)
            losses=runner.model(**processed,mode='loss');loss,_=runner.model.parse_losses({k:v for k,v in losses.items() if k!='loss_camera_aux'});loss.backward()
            f=runner.model.fusion_module
            groups=dict(camera=runner.model.img_backbone.parameters(),bottleneck=[f.bottleneck_tokens])
            if f.use_bottleneck_gate:groups['gates']=f.bottleneck_gates.parameters()
            for name,params in groups.items():
                grads=[p.grad for p in params if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().max()>0 for g in grads),name
            runner.model.zero_grad(set_to_none=True);f.camera_mode='missing';seen=[]
            def identity(m,inputs,output):
                assert torch.equal(inputs[1],output['bev_features']);seen.append(True)
            handle=f.register_forward_hook(identity)
            with torch.no_grad():pred=runner.model(**processed,mode='predict')
            handle.remove();assert seen
            assert all(torch.isfinite(x.pred_instances_3d.scores_3d).all() for x in pred)
            report.update(status='passed',strict_reload=True,optimizer_scheduler_saved=True,
                detection_only_gradient_groups=list(groups),missing_feature_identity=True,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as e:
        report.update(status='failed',error=repr(e));raise
    finally:write(dest/('preflight.json' if smoke else 'study_progress.json'),report)

def queue(args):
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    entries=plan(root);state=dict(status='planned',stages=entries,scope='Full KITTI and nuScenes sweeps',
        reused_reference='Completed gated B8/E256 on each dataset; no retraining of those arms')
    old=root/'status.json'
    if old.exists():
        previous=json.loads(old.read_text());byname={x['name']:x for x in previous['stages']}
        for x in entries:
            prior=byname[x['name']];assert prior['config_sha256']==x['config_sha256']
            if prior['status']=='complete':x.update(prior)
    write(root/'status.json',state)
    if args.mode=='plan':return
    try:
        state['status']='running'
        for e in entries:
            if e['status']=='complete':continue
            state['current_stage']=e['name'];dest=Path(e['work_dir']);pf=root/'preflights'/e['name']
            passed=pf/'preflight.json'
            good=[]
            for candidate in [pf]+sorted(pf.parent.glob(pf.name+'_attempt*')):
                record=candidate/'preflight.json'
                if record.exists():
                    r=json.loads(record.read_text())
                    if r.get('status')=='passed' and r.get('production_config_sha256')==e['config_sha256']:good.append(record)
            if good:passed=good[-1]
            if not good:
                e['status']='preflight';write(root/'status.json',state)
                # A failed preflight is preserved; retry in a separate numbered directory.
                original=pf;attempt=1
                while pf.exists():pf=original.with_name(original.name+f'_attempt{attempt}');attempt+=1
                passed=pf/'preflight.json'
                command=[sys.executable,__file__,'preflight','--config',e['config'],'--dataset',e['dataset'],'--output',str(pf)]
                with (root/(e['name']+'_preflight.log')).open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads(passed.read_text());assert report['status']=='passed'
            assert report['production_config_sha256']==e['config_sha256']
            e.update(status='training',preflight=str(passed));write(root/'status.json',state)
            command=[sys.executable,__file__,'train','--config',e['config'],'--dataset',e['dataset'],'--preflight',str(passed)]
            with (root/(e['name']+'.log')).open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            progress=json.loads((dest/'study_progress.json').read_text());assert progress['status']=='complete'
            e.update(status='complete',progress=progress);write(root/'status.json',state)
        state['status']='complete'
    except BaseException as e:
        state.update(status='failed',error=repr(e));raise
    finally:write(root/'status.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['plan','queue','preflight','train'])
    p.add_argument('--root');p.add_argument('--config');p.add_argument('--dataset',choices=['kitti','nuscenes']);p.add_argument('--output');p.add_argument('--preflight')
    a=p.parse_args()
    if a.mode in ('plan','queue'):queue(a)
    else:run_arm(a)
