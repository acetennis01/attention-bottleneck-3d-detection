"""Separate, untested full-run LR comparison; do not launch alongside the main run."""
_base_ = ['./nuscenes_multiscale_fresh.py']
optim_wrapper = dict(optimizer=dict(lr=1e-4))
# The inherited relative cosine minimum keeps the entire LR curve threefold
# lower, including every component group, without another schedule change.
work_dir = 'work_dirs/nuscenes_multiscale_fresh_r50_e256_b8_lr1e4_seed0'
val_evaluator = dict(jsonfile_prefix=work_dir + '/validation')
test_evaluator = val_evaluator
