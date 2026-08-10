
# python tools/eval_from_pkl.py \
#   configs/my_model/my_config.py \
#   work_dirs/my_model/pred_instances_3d.pkl \
#   --metrics 3d bbox bev

import os
import pickle
import mmengine
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS, METRICS
# from mmengine.dataset import build_dataloader
from mmdet3d.utils import register_all_modules
import argparse

from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate prediction results from pkl')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('pkl', help='Path to pkl result file')
    parser.add_argument('--metrics', nargs='+', help='Evaluation metrics, e.g., bbox bev 3d')
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help='Custom options')
    return parser.parse_args()


def main():
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    # Build dataset
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    dataset.test_mode = True

    # Load predictions from pkl
    print(f'Loading predictions from: {args.pkl}')
    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)
    print("results pkl", results[0])
    # Build metric
    evaluator = METRICS.build({
        **cfg.test_evaluator,
        'metric': args.metrics,
    })


    for i, pred in enumerate(results):
        sample = Det3DDataSample()
        instance = InstanceData()
        instance.bboxes_3d = pred['boxes_3d']
        instance.labels_3d = pred['labels_3d']
        instance.scores_3d = pred['scores_3d']
        sample.pred_instances_3d = instance
        wrapped_results.append(sample)

    # Set metainfo (important!)
    evaluator.dataset_meta = dataset.metainfo

    # Feed each result to the evaluator
    for i in range(len(dataset)):
        evaluator.process(None, [wrapped_results[i]])

    # Run evaluation
    eval_results = evaluator.evaluate()
    print('\nEvaluation results:')
    for k, v in eval_results.items():
        print(f'{k}: {v}')

if __name__ == '__main__':
    main()