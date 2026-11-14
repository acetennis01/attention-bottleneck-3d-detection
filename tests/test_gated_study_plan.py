from copy import deepcopy
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('study_plan',Path(__file__).parents[1]/'tools/gated_study_plan.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_full_sweeps_have_expected_independent_interventions():
    for dataset in ('kitti','nuscenes'):
        arms=m.study_arms(dataset)
        assert len(arms)==9 and len({x['name'] for x in arms})==9
        assert [a['changes']['num_bottleneck_tokens'] for a in arms if a['name'].startswith('b')]==[1,2,4,16]
        assert [a['changes']['embed_dim'] for a in arms if 'width' in a['name']]==[128,512]

def test_changes_leave_all_training_and_backbone_settings_untouched():
    for dataset in ('kitti','nuscenes'):
        fusion=dict(use_bottleneck_gate=True,camera_drop_prob=.2,
            num_bottleneck_tokens=8,embed_dim=256)
        if dataset=='nuscenes':fusion['camera_view_drop_prob']=.15
        base=dict(model=dict(fusion_module=fusion,img_backbone=dict(type='fixed')),
            optim_wrapper=dict(lr=1e-4),train_cfg=dict(budget=40),data=dict(seed=0))
        for arm in m.study_arms(dataset):
            before=deepcopy(base);cfg=m.apply_arm(base,arm)
            assert base==before
            assert cfg['model']['img_backbone']==base['model']['img_backbone']
            assert cfg['optim_wrapper']==base['optim_wrapper']
            if arm['name']=='gate_no_dropout':
                assert cfg['model']['fusion_module']['use_bottleneck_gate']
                assert cfg['model']['fusion_module']['camera_drop_prob']==0
                if dataset=='nuscenes':assert cfg['model']['fusion_module']['camera_view_drop_prob']==0


def test_cold_ablation_cannot_load_a_saved_fusion_resume_checkpoint():
    old=dict(resume=True,load_from='/old/fusion/iter_158000.pth',model=dict(x=1))
    new=m.cold_start_reference(old,'nuscenes')
    assert new['resume'] is False and new['load_from'] is None
    assert new['model']==old['model'] and old['resume'] is True
    lidar=m.cold_start_reference(dict(resume=True,load_from='lidar_epoch28.pth'),'kitti')
    assert lidar['load_from']=='lidar_epoch28.pth' and lidar['resume'] is False


def test_recovery_ignores_unaligned_and_weights_only_checkpoints(tmp_path):
    for name in ['epoch_2.pth','epoch_4.pth','epoch_6.pth','aligned_epoch_4.pth',
                 'best_metric_epoch_8.pth']:
        (tmp_path/name).write_bytes(b'test')
    selected=m.aligned_kitti_checkpoints(tmp_path)
    assert [x.name for x in selected]==['epoch_4.pth','aligned_epoch_4.pth']
    (tmp_path/'epoch_8.pth').write_bytes(b'test')
    assert m.aligned_kitti_checkpoints(tmp_path)[-1].name=='epoch_8.pth'
