"""One change versus frozen: reduce independent camera aux weight .25 -> .05."""
_base_ = ['./nuscenes_controlled_base.py']
model = dict(camera_aux_loss_weight=0.05)
work_dir = 'work_dirs/nuscenes_controlled_20261021/lower_aux'
val_evaluator = dict(jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
