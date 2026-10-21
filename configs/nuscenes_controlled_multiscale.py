"""One architectural change versus frozen: MBT readout also at FPN levels 1/2."""
_base_ = ['./nuscenes_controlled_base.py']
model = dict(type='NuScenesMultiScaleMBTDetectorV2',
             fusion_module=dict(type='NuScenesMultiScaleMBTV2'))
# Prepared by tools/run_nuscenes_controlled.py; all old tensors are identical
# to the reference and only additional query/zero-output adapters are added.
load_from = ('work_dirs/nuscenes_controlled_20261021/'
             'multiscale_initializer.pth')
work_dir = 'work_dirs/nuscenes_controlled_20261021/multiscale'
val_evaluator = dict(jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
