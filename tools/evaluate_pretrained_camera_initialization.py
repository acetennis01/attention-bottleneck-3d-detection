"""Full nuScenes validation of BOTH pretrained encoders at zero MBT output.

This verifies initialization safety, not camera-only detection performance.
The backbone has no 3D head and the new fusion has not learned camera use yet.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules

from projects.myfusion.tools.transfusion_l_checkpoint import (
    load_lidar_checkpoint, sha256_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', default='work_dirs/transfusion_l_mbt_swint_initialization')
    args = parser.parse_args()
    root = Path(args.work_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite evaluation: {root}')
    baseline_file = Path('work_dirs/transfusion_l_pretrained_baseline/baseline_verification.json')
    baseline = json.loads(baseline_file.read_text())
    if baseline['status'] != 'passed' or baseline['validation_frames'] != 6019:
        raise RuntimeError('Reproduce LiDAR baseline first')
    root.mkdir(parents=True, exist_ok=True)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile('projects/myfusion/configs/transfusion_l_mbt_swint_nuscenes.py')
    cfg.work_dir = str(root)
    cfg.test_evaluator.jsonfile_prefix = str(root / 'results')
    cfg.load_from = None
    # This is a TestLoop only; no optimizer steps or training hooks run.
    runner = Runner.from_cfg(cfg)
    assert len(runner.test_dataloader.dataset) == 6019
    assert sha256_file(cfg.test_evaluator.ann_file) == baseline['annotation_sha256']
    transfer = load_lidar_checkpoint(runner.model, cfg.lidar_checkpoint, fusion=True)
    camera = runner.model.img_backbone.pretrained_verification
    assert camera['sha256'] == cfg.camera_checkpoint_sha256
    for tensor in runner.model.fusion_module.bev_residual.parameters():
        assert torch.count_nonzero(tensor) == 0
    report = dict(status='running', kind='zero_residual_initialization_only',
        started_at=datetime.now().astimezone().isoformat(),
        pretrained_lidar=transfer, pretrained_camera=camera,
        validation_frames=6019, optimizer_updates=0,
        reference_measured_lidar=baseline['measured'], tolerance_raw=.0001,
        standalone_camera_mAP=None,
        caveat='No camera-only 3D head is present. Matching LiDAR scores verifies '
               'baseline preservation; it does not demonstrate camera improvement.')
    path = root / 'camera_initialization_verification.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        runner.test()
        metrics_file = root / 'results/pred_instances_3d/metrics_summary.json'
        metrics = json.loads(metrics_file.read_text())
        predictions = json.loads((metrics_file.parent / 'results_nusc.json').read_text())
        assert len(predictions['results']) == 6019
        report['measured'] = dict(mAP=metrics['mean_ap'], NDS=metrics['nd_score'])
        report['differences_from_lidar'] = {key: value - baseline['measured'][key]
                                          for key, value in report['measured'].items()}
        passed = all(abs(value) <= report['tolerance_raw']
                     for value in report['differences_from_lidar'].values())
        report.update(status='passed' if passed else 'mismatch',
                      completed_at=datetime.now().astimezone().isoformat())
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            raise RuntimeError('Zero-output MBT did not preserve baseline within 0.01 percentage points')
    except BaseException as error:
        if report['status'] == 'running':
            report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
