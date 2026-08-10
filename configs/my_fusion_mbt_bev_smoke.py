"""Two-epoch checkpointed diagnostic run for MBT BEV fusion on KITTI."""

_base_ = ['./my_fusion_mbt_bev.py']

work_dir = 'work_dirs/mbt_bev_diagnostic'

train_dataloader = dict(batch_size=1)
train_cfg = dict(by_epoch=True, max_epochs=2, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=2, save_last=True))

# KittiMetric appends ``pred_instances_3d.pkl`` beneath this prefix.  Unlike
# the default TemporaryDirectory, these predictions survive evaluation.
val_evaluator = dict(pklfile_prefix=work_dir)
test_evaluator = dict(pklfile_prefix=work_dir)
