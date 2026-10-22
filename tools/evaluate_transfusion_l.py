"""Strict-load and evaluate the official LiDAR baseline before MBT training.

Run from the MMDetection3D root with PYTHONPATH pointing to that root.
Full validation only: a partial run is not a baseline reproduction.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules

from projects.myfusion.tools.transfusion_l_checkpoint import (
    load_lidar_checkpoint, sha256_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='projects/myfusion/configs/transfusion_l_nuscenes_pretrained.py')
    parser.add_argument('--work-dir')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    directory = Path(args.work_dir or cfg.work_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Refusing to overwrite evaluation: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(directory)
    cfg.test_evaluator.jsonfile_prefix = str(directory / 'results')
    # The audited loader below also handles the explicit spconv2 -> legacy
    # sparse-kernel storage permutation; don't first do a permissive raw load.
    cfg.load_from = None
    register_all_modules(init_default_scope=True)
    runner = Runner.from_cfg(cfg)
    assert len(runner.test_dataloader.dataset) == 6019
    runner.load_or_resume()
    transfer = load_lidar_checkpoint(runner.model, cfg.lidar_checkpoint)
    report = dict(status='running', started_at=datetime.now().astimezone().isoformat(),
        checkpoint=transfer, validation_frames=6019,
        annotation_sha256=sha256_file(cfg.test_evaluator.ann_file),
        upstream_mmdetection3d_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        published_reference=dict(mAP=.649, NDS=.696),
        tolerance_raw=.005,
        note='OpenMMLab model-zoo reference, not the separate MIT checkpoint. '
             'Deterministic nearest-nine-sweep validation is used for both baseline and MBT.')
    report_path = directory / 'baseline_verification.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        runner.test()
        metrics_file = directory / 'results/pred_instances_3d/metrics_summary.json'
        metrics = json.loads(metrics_file.read_text())
        predictions = json.loads((metrics_file.parent / 'results_nusc.json').read_text())
        assert len(predictions['results']) == 6019
        report['measured'] = dict(mAP=metrics['mean_ap'], NDS=metrics['nd_score'])
        report['reference_differences'] = {
            key: report['measured'][key] - target
            for key, target in report['published_reference'].items()}
        passed = all(abs(delta) <= report['tolerance_raw']
                     for delta in report['reference_differences'].values())
        report.update(status='passed' if passed else 'mismatch',
                      completed_at=datetime.now().astimezone().isoformat())
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            raise RuntimeError('Baseline does not reproduce within 0.5 percentage points. Diagnose before fusion training.')
    except BaseException as error:
        if report['status'] == 'running':
            report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
