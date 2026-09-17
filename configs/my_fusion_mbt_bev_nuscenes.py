"""Camera-focused six-view MBT detector on nuScenes v1.0 trainval.

This follows the final KITTI training methodology: warm-start from the best
matched LiDAR-only detector, freeze the pretrained LiDAR and detection path
for five epochs, and then fine-tune it more gently than the camera/fusion
path.
"""

from glob import glob as _glob
from os.path import getmtime as _getmtime

_base_ = ['./my_fusion_mbt_bev_nuscenes_mini.py']

custom_imports = dict(
    imports=[
        'projects.myfusion.fusion.mbt_bev',
        'projects.myfusion.fusion.mbt_detector',
        'projects.myfusion.fusion.mbt_bev_nuscenes',
        'projects.myfusion.fusion.mbt_detector_nuscenes',
        'projects.myfusion.hooks.fusion_diagnostics',
        'projects.myfusion.hooks.staged_fusion_training',
    ],
    allow_failed_imports=False)

data_root = 'data/nuscenes_full/'
metainfo = dict(classes=_base_.class_names, version='v1.0-trainval')

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        metainfo=metainfo))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    jsonfile_prefix='work_dirs/mbt_bev_nuscenes/results')
test_evaluator = val_evaluator

# CheckpointHook keeps exactly one best-NDS file. Resolve it when this config
# is loaded so a later epoch can replace the current best without requiring a
# hard-coded epoch number here. The fallback path deliberately fails clearly
# if training is launched before the LiDAR baseline has produced a best file.
_lidar_best_candidates = _glob(
    'work_dirs/pointpillars_nuscenes/'
    'best_NuScenes*NuScenes_NDS_epoch_*.pth')
load_from = (
    max(_lidar_best_candidates, key=_getmtime)
    if _lidar_best_candidates else
    'work_dirs/pointpillars_nuscenes/BEST_LIDAR_CHECKPOINT_NOT_FOUND.pth')

custom_hooks = [
    dict(type='StagedFusionTrainingHook', freeze_epochs=5),
    dict(type='FusionDiagnosticsHook', interval=500),
]

# Preserve the nuScenes effective batch size of four. During fine-tuning, the
# camera/fusion path learns four times faster than the pretrained LiDAR path.
epoch_num = 24
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=epoch_num,
    val_begin=2,
    val_interval=1)
optim_wrapper = dict(
    accumulative_counts=4,
    optimizer=dict(
        type='AdamW',
        lr=3e-4,
        betas=(0.95, 0.99),
        weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=2.0),
            'fusion_module': dict(lr_mult=2.0),
            'voxel_encoder': dict(lr_mult=0.5),
            'middle_encoder': dict(lr_mult=0.5),
            'backbone': dict(lr_mult=0.5),
            'neck': dict(lr_mult=0.5),
            'bbox_head': dict(lr_mult=0.5),
        }))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num,
        eta_min=3e-6,
        begin=0,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
]

work_dir = 'work_dirs/mbt_bev_nuscenes'
