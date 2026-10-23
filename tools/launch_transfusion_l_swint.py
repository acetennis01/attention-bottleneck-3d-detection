"""Preflight and exec a FRESH full Swin/TransFusion-L MBT training run.

Use standard train.py --resume for later resumes, not this cold-start launcher.
Does not use smoke-trained weights or previous PointPillars fusion weights.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from mmengine.config import Config
from projects.myfusion.tools.transfusion_l_checkpoint import sha256_file, SHA256
from projects.myfusion.fusion.nuimages_swin import NUIMAGES_SWIN_SHA256


def assert_no_other_job():
    scripts = {'train.py', 'test.py', 'evaluate_transfusion_l.py',
               'evaluate_pretrained_camera_initialization.py',
               'smoke_transfusion_l_runner.py', 'run_ablations.py'}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            arguments = (process / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        if arguments and 'python' in Path(arguments[0]).name:
            if any(Path(argument).name in scripts for argument in arguments[1:] if argument):
                raise RuntimeError(f'Another training/evaluation job is active: {process.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--smoke-report', default='work_dirs/transfusion_l_swint_runner_smoke_20261022b/report.json')
    args = parser.parse_args()
    assert_no_other_job()
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads(Path(args.source_manifest).read_text())
    for name, checksum in manifest['python_sha256'].items():
        if sha256_file(project / name) != checksum:
            raise RuntimeError(f'PC source differs from verified branch: {name}')
    filename = 'projects/myfusion/configs/transfusion_l_mbt_swint_nuscenes.py'
    cfg = Config.fromfile(filename)
    assert cfg.resume is False and cfg.load_from is None
    assert cfg.train_cfg.max_iters == 160000
    assert cfg.optim_wrapper.accumulative_counts == 4
    assert cfg.model.freeze_lidar is True
    assert cfg.train_dataloader.dataset.get('indices') is None
    assert cfg.val_dataloader.dataset.get('indices') is None
    assert cfg.val_evaluator.get('format_only', False) is False
    assert len(cfg.class_names) == 10
    assert sha256_file(cfg.lidar_checkpoint) == SHA256
    assert sha256_file(cfg.camera_checkpoint) == NUIMAGES_SWIN_SHA256
    baseline_path = Path('work_dirs/transfusion_l_pretrained_baseline/baseline_verification.json')
    camera_path = Path('work_dirs/transfusion_l_mbt_swint_initialization/camera_initialization_verification.json')
    baseline = json.loads(baseline_path.read_text())
    camera = json.loads(camera_path.read_text())
    smoke = json.loads(Path(args.smoke_report).read_text())
    assert baseline['status'] == camera['status'] == smoke['status'] == 'passed'
    assert baseline['validation_frames'] == camera['validation_frames'] == 6019
    assert sha256_file(cfg.val_evaluator.ann_file) == baseline['annotation_sha256']
    assert smoke['exact_model_optimizer_scheduler_restore_at_iter_8']
    assert smoke['total_optimizer_updates'] == 4
    assert smoke['lidar_and_head_parameters_and_bn_buffers_unchanged']
    work_dir = Path(cfg.work_dir)
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f'Refusing to restart over existing training: {work_dir}')
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    record = dict(status='preflight_passed', launched_at=datetime.now().astimezone().isoformat(),
        training_pid=os.getpid(), training_process_group=os.getpgid(0),
        config=filename, work_dir=str(work_dir.resolve()),
        verified_source_commit=manifest['verified_source_commit'],
        source_python_files_verified=len(manifest['python_sha256']),
        pc_project_git_head=subprocess.check_output(
            ['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip(),
        source_commit_verified_by_file_hashes_not_pc_HEAD=True,
        lidar_checkpoint_sha256=SHA256, camera_checkpoint_sha256=NUIMAGES_SWIN_SHA256,
        full_baseline=baseline['measured'], full_camera_initialization=camera['measured'],
        smoke_report=str(Path(args.smoke_report).resolve()), smoke_weights_used=False,
        fresh_fusion=True, resume=False, seed=cfg.randomness.seed,
        microbatch_budget=160000, optimizer_update_budget=40000,
        first_validation_microbatch=2000, later_validation_interval=10000)
    (work_dir / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)
    os.execv(sys.executable, [sys.executable, '-u', 'tools/train.py', filename])


if __name__ == '__main__':
    main()
