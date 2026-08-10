_base_ = '/home/abhiramannaluru/mmdetection3d/projects/myfusion/configs/my_fusion_mbt_token.py'
default_scope = 'mmdet3d'

test_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(
            type='KittiMetric',
            # If your base cfg already sets `data_root`, you can keep this relative:
            ann_file='/home/abhiramannaluru/data/kitti/kitti_infos_val.pkl',
            pklfile_prefix='/home/abhiramannaluru/mmdetection3d/work_dirs/myfusion_eval/results',
            format_only=False,
        )
    ]
)
