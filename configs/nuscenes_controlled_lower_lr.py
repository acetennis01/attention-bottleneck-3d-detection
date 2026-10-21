"""One change versus frozen: reduce every trainable LR by threefold."""
_base_ = ['./nuscenes_controlled_base.py']
optim_wrapper = dict(optimizer=dict(lr=1e-4))
work_dir = 'work_dirs/nuscenes_controlled_20261021/lower_lr'
val_evaluator = dict(jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
