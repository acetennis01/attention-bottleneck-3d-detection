"""Four fixed-checkpoint full-validation conditions; inference only."""
import fcntl,gc,hashlib,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from mmengine.config import Config
from projects.myfusion.tools.transfusion_l_checkpoint import state_for_model,sha256_file,SHA256

BASE=Path('/home/yab/mmdetection3d')
ROOT=BASE/'work_dirs/part_b_run3_camera_eval_20261109'
RUN=BASE/'work_dirs/part_b_20261107_transfusion_gated_full'
CONFIG=BASE/'work_dirs/part_b_20261107_training_queue/transfusion_gated_full.py'
CP=RUN/'best_NuScenes metric_pred_instances_3d_NuScenes_NDS_iter_110000.pth'
REFERENCE=json.loads((ROOT/'reference.json').read_text())
state=dict(status='preflight',selected_microbatch=110000,inference_only=True,
           checkpoint=str(CP),config=str(CONFIG),cases={})

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def progress():write(ROOT/'status.json',state)

def summarize():
    rows=[]
    clean=state['cases'].get('correct',{}).get('metrics')
    previous={r['mode']:r for r in REFERENCE['previous_results_percentage']}
    for mode,case in state['cases'].items():
        if case['status']!='passed':continue
        metrics=case['metrics']; row=dict(mode=mode,mAP=metrics['mAP']*100,NDS=metrics['NDS']*100)
        if clean:
            row['clean_minus_condition_pp']={k:100*(clean[k]-metrics[k]) for k in ('mAP','NDS')}
        if mode in previous:
            row['new_minus_previous_pp']={k:row[k]-previous[mode][k] for k in ('mAP','NDS')}
            row['previous_clean_minus_condition_pp']={k:previous['correct'][k]-previous[mode][k] for k in ('mAP','NDS')}
        rows.append(row)
    write(ROOT/'comparison.json',dict(status=state['status'],checkpoint_sha256=state.get('checkpoint_sha256'),
          selected_microbatch=110000,units='percentages and percentage-point differences',results=rows,
          previous_reference=REFERENCE,caveat='Single seed. Black/shuffle are OOD camera-dependence tests; explicit missing uses an oracle availability signal, not automatic corruption detection.'))

lock=(ROOT/'evaluation_queue.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    progress()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:args=(d/'cmdline').read_bytes().decode().split('\0')
        except (OSError,UnicodeDecodeError):continue
        if args and 'python' in Path(args[0]).name and any(Path(a).name in ('train.py','evaluate_nuscenes_camera_case.py','resume_part_b_run3_158000_20261109.py') for a in args[1:] if a):
            raise RuntimeError('Another trainer/evaluator is active: '+d.name)
    cfg=Config.fromfile(CONFIG)
    assert sha256_file(CONFIG)=='47000ad8bb2728a17b2f89a7fd8b66e94b99f5bb87aed8d678854807bd71d489'
    assert cfg.model.freeze_lidar and cfg.model.fusion_module.camera_mode=='present'
    state['checkpoint_sha256']=sha256_file(CP)
    state['config_sha256']=sha256_file(CONFIG)
    selected=torch.load(CP,map_location='cpu')
    assert selected['meta']['iter']==110000
    state['checkpoint_metadata_iteration']=selected['meta']['iter']
    initializer=BASE/cfg.lidar_checkpoint
    assert sha256_file(initializer)==SHA256
    raw=torch.load(initializer,map_location='cpu')['state_dict']
    converted,_=state_for_model(SimpleNamespace(state_dict=lambda:selected['state_dict']),raw)
    for key,value in converted.items():
        assert torch.equal(selected['state_dict'][key],value),key
    state['protected_lidar_tensors_exactly_verified']=len(converted)
    state['annotation_sha256']=sha256_file(BASE/cfg.test_evaluator.ann_file)
    expected=json.loads((RUN/'validation/iter_110000/pred_instances_3d/metrics_summary.json').read_text())
    del selected,raw,converted;gc.collect()
    state['status']='running';progress()
    for mode in ('correct','black','shuffled','missing'):
        destination=ROOT/mode
        if destination.exists():raise FileExistsError(destination)
        state['active_case']=mode;state['cases'][mode]=dict(status='running',work_dir=str(destination));progress()
        with (ROOT/(mode+'.log')).open('x') as log:
            subprocess.run([sys.executable,'projects/myfusion/tools/evaluate_nuscenes_camera_case.py',
                '--config',str(CONFIG),'--checkpoint',str(CP),'--mode',mode,'--work-dir',str(destination)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((destination/'metrics.json').read_text())
        assert report['status']=='passed' and report['validation_samples']==6019
        assert report['checkpoint_sha256']==state['checkpoint_sha256']
        assert report['annotation_sha256']==state['annotation_sha256']
        state['cases'][mode].update(status='passed',metrics=report['metrics'],report=str(destination/'metrics.json'))
        if mode=='correct':
            # Tolerance is .001 percentage points, fixed before evaluation.
            for metric,key in (('mAP','mean_ap'),('NDS','nd_score')):
                assert abs(report['metrics'][metric]-expected[key])<=1e-5,'Clean checkpoint reevaluation differs'
        if mode=='missing':
            assert report['exact_lidar_feature_identity_batches']==6019
            for metric,key in (('mAP','mean_ap'),('NDS','nd_score')):
                assert abs(report['metrics'][metric]-REFERENCE['lidar_baseline_unit_scale'][key])<=1e-5,'Fallback differs from baseline'
        progress();summarize()
    assert sha256_file(CP)==state['checkpoint_sha256']
    state['status']='complete';state.pop('active_case',None);progress();summarize()
except BaseException as error:
    state.update(status='failed',error=repr(error));progress();summarize();raise
