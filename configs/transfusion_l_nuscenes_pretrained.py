"""Unmodified OpenMMLab TransFusion-L model; local data/output overrides only.

This is the MMDetection3D model-zoo checkpoint (64.9 mAP / 69.6 NDS),
not MIT's separate 64.68 / 69.28 checkpoint. Reproduce before fusion training.
"""
import os

_base_ = ['../../BEVFusion/configs/'
          'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

data_root = os.environ.get('NUSCENES_DATA_ROOT', 'data/nuscenes_full/').rstrip('/') + '/'
metainfo = dict(classes=_base_.class_names, version='v1.0-trainval')
work_dir = 'work_dirs/transfusion_l_pretrained_baseline'
lidar_checkpoint = ('checkpoints/transfusion_l/'
    'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth')
lidar_checkpoint_sha256 = '2628f933277774e194932815ea1a5d66a0bd9f69ebc8d9040e8c92760bb6652e'
load_from = lidar_checkpoint
resume = False

# Model, voxelizer, point features, nine sweeps, class order and range are
# inherited exactly. In particular, do not reuse PointPillars' 4D features,
# ten historical sweeps, [-50,50] range, or three-level head.
test_pipeline = [dict(step) for step in _base_.test_pipeline]
test_pipeline[1]['test_mode'] = True
# Pin nearest-nine sweeps, rather than randomly selecting nine at evaluation.
# This is the only algorithmic eval override; use it identically for MBT.
val_dataloader = dict(
    num_workers=2, persistent_workers=False,
    dataset=dict(data_root=data_root, metainfo=metainfo, pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
randomness = dict(seed=0, deterministic=False, diff_rank_seed=False)

# Evaluation-only: accidentally invoking train.py must not retrain the model.
train_dataloader = None
train_cfg = None
optim_wrapper = None
param_scheduler = None
