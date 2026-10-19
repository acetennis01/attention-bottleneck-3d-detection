"""Only bottleneck count/output paths differ from the eight-token V2 run."""

_base_ = ['./nuscenes_camera_v2.py']

model = dict(fusion_module=dict(num_bottleneck_tokens=16))
work_dir = 'work_dirs/nuscenes_camera_v2_r50_e256_b16_seed0'
val_evaluator = dict(jsonfile_prefix=work_dir + '/results')
test_evaluator = val_evaluator
