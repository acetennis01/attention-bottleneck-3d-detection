"""Replay disposable LiDAR quality-head optimization controls from a cache.

No detector training or detection AP evaluation is performed. The input cache
and guided checkpoint must match the completed diagnosis record.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from projects.myfusion.tools.diagnose_sensor_reliability import fresh_head, refit
from projects.myfusion.tools.run_gated_studies import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--diagnosis-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    torch.set_num_threads(2)
    diagnosis = json.loads((args.diagnosis_dir / 'diagnosis.json').read_text())
    cache_path = args.diagnosis_dir / 'feature_cache.pt'
    checkpoint = args.study / 'guided/epoch_6.pth'
    assert diagnosis['status'] == 'complete'
    assert sha(cache_path) == diagnosis['feature_cache_sha256']
    assert sha(checkpoint) == diagnosis['checkpoint_sha256']
    cache = torch.load(cache_path, map_location='cpu')
    state = torch.load(checkpoint, map_location='cpu')['state_dict']
    prefix = 'fusion_module.lidar_reliability.'
    head = fresh_head(cache['train']['lidar'].shape[1])
    head.load_state_dict({key.removeprefix(prefix): value
                          for key, value in state.items() if key.startswith(prefix)},
                         strict=True)
    reset = deepcopy(head)
    torch.nn.init.zeros_(reset[-1].weight)
    torch.nn.init.zeros_(reset[-1].bias)
    report = dict(status='complete',
                  note='Disposable LiDAR head refits on the identical frozen feature cache; no detector update.')
    for name, initial, objective in (
            ('trained_l1_lr1e4', head, 'l1'),
            ('trained_bce_lr1e4', head, 'bce'),
            ('reset_half_l1_lr1e4', reset, 'l1')):
        report[name] = refit(initial, cache['train'], cache['validation'],
                             'lidar', 1, objective, learning_rate=1e-4)
    assert sha(checkpoint) == diagnosis['checkpoint_sha256']
    write(args.output, report)


if __name__ == '__main__':
    main()
