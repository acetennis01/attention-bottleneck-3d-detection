"""Sequential full-split ablations of completed KITTI and nuScenes models.

Run on the GPU host from the MMDetection3D root. Original weights, configs and
datasets are read-only. Each case has an exclusive directory and separate log.
An interrupted case is retried in a NEW attempt directory with --resume; passed
cases are reused only after checking their checkpoint hash and frame count.
"""
import argparse
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


MODES = ('correct', 'black', 'shuffled')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now().astimezone().isoformat()


def model_specs(root):
    # Latest nuScenes model first; omit failed experiments and disposable pilots.
    names = [('nuscenes_transfusion_swint_b8', 'nuscenes',
              'transfusion_l_mbt_swint_e256_b8_seed0', 'transfusion_l_mbt_swint_nuscenes.py'),
             ('kitti_v2_b8', 'kitti', 'kitti_camera_v2_r50_e256_b8_seed0', 'kitti_camera_v2.py'),
             ('kitti_v2_b16', 'kitti', 'kitti_camera_v2_r50_e256_b16_seed0', 'kitti_camera_v2_b16.py'),
             ('nuscenes_multiscale_b8', 'nuscenes',
              'nuscenes_multiscale_fresh_r50_e256_b8_seed0', 'nuscenes_multiscale_fresh.py'),
             ('nuscenes_original_b4', 'nuscenes',
              'mbt_bev_nuscenes_fixed', 'my_fusion_mbt_bev_nuscenes.py')]
    names += [(f'kitti_legacy_b{tokens}', 'kitti',
               f'kitti_bottleneck_b{tokens}_valid_only_seed0', f'kitti_bottleneck_b{tokens}.py')
              for tokens in (1, 2, 4, 8, 16)]
    models = []
    for label, dataset, directory, filename in names:
        run = root / 'work_dirs' / directory
        choices = list(run.glob('best*.pth'))
        if len(choices) != 1:
            raise RuntimeError(f'{label}: expected one selected best checkpoint, found {choices}')
        config = run / filename  # Saved resolved config from this actual training.
        if not config.is_file():
            raise FileNotFoundError(config)
        models.append(dict(name=label, dataset=dataset, config=str(config),
                           config_sha256=digest(config), checkpoint=str(choices[0]),
                           checkpoint_sha256=digest(choices[0]),
                           validation_samples=3769 if dataset == 'kitti' else 6019))
    return models


def no_other_job():
    scripts = {'train.py', 'test.py', 'evaluate_kitti_camera_case.py',
               'evaluate_nuscenes_camera_case.py', 'run_ablations.py',
               'evaluate_transfusion_l.py', 'evaluate_pretrained_camera_initialization.py'}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            args = (directory / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        if args and 'python' in Path(args[0]).name and any(
                Path(arg).name in scripts for arg in args[1:] if arg):
            raise RuntimeError(f'Another trainer/evaluator is active: {directory.name}')


def metrics_for(model, report):
    if model['dataset'] == 'nuscenes':
        return {key: 100 * report['metrics'][key] for key in ('mAP', 'NDS')}
    values = {}
    for key, suffix in [('3D_AP40_moderate', 'Overall_3D_AP40_moderate'),
                        ('BEV_AP40_moderate', 'Overall_BEV_AP40_moderate')]:
        matches = [value for name, value in report['metrics'].items()
                   if name.endswith('/' + suffix)]
        if len(matches) != 1:
            raise RuntimeError(f'Missing unique metric {suffix}')
        values[key] = matches[0]
    return values


def summarize(output, state):
    rows, differences = [], []
    for model in state['models']:
        conditions = {}
        for mode in MODES:
            case = state['cases'].get(model['name'] + '/' + mode, {})
            if case.get('status') != 'passed':
                continue
            report = json.loads(Path(case['report']).read_text())
            conditions[mode] = metrics_for(model, report)
            rows.append(dict(model=model['name'], dataset=model['dataset'], mode=mode,
                             **conditions[mode]))
        for mode in ('black', 'shuffled'):
            if 'correct' in conditions and mode in conditions:
                differences.append(dict(model=model['name'], dataset=model['dataset'],
                    corruption=mode, **{key: value - conditions[mode][key]
                    for key, value in conditions['correct'].items()}))
    caveat = ('Black/shuffled images test sensitivity to out-of-distribution camera '
              'corruptions. A drop supports camera dependence, not statistical '
              'significance or attribution of the entire gain over a LiDAR baseline. '
              'Scores and differences below use percentage/AP points.')
    write(output / 'comparison.json', dict(results=rows,
        correct_minus_corrupted=differences, caveat=caveat))
    fields = ['model', 'dataset', 'mode', 'mAP', 'NDS', '3D_AP40_moderate', 'BEV_AP40_moderate']
    with (output / 'overall_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Camera ablation evaluations', '', caveat, '',
             '| Model | Camera | mAP / 3D AP40 moderate | NDS / BEV AP40 moderate |',
             '|---|---|---:|---:|']
    for row in rows:
        a, b = (('mAP', 'NDS') if row['dataset'] == 'nuscenes'
                else ('3D_AP40_moderate', 'BEV_AP40_moderate'))
        lines.append(f"| {row['model']} | {row['mode']} | {row[a]:.4f} | {row[b]:.4f} |")
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = Path(args.work_dir).resolve()
    path = output / 'suite.json'
    no_other_job()
    if args.resume:
        state = json.loads(path.read_text())
    else:
        if output.exists():
            raise FileExistsError(output)
        models = model_specs(root)
        output.mkdir(parents=True)
        state = dict(status='planned', started_at=now(), models=models, cases={},
                     total_cases=len(models) * len(MODES), active_case=None,
                     queue_pid=os.getpid(), inference_only=True, shuffle_seed=0)
        write(path, state)
    if args.plan_only:
        print(json.dumps(state, indent=2), flush=True)
        return
    state.update(status='running', queue_pid=os.getpid())
    child = None
    try:
        for model in state['models']:
            for mode in MODES:
                key = model['name'] + '/' + mode
                previous = state['cases'].get(key, {})
                if digest(model['checkpoint']) != model['checkpoint_sha256']:
                    raise RuntimeError(f'Checkpoint changed: {key}')
                if digest(model['config']) != model['config_sha256']:
                    raise RuntimeError(f'Source config changed: {key}')
                if previous.get('status') == 'passed':
                    report = json.loads(Path(previous['report']).read_text())
                    assert report['checkpoint_sha256'] == model['checkpoint_sha256']
                    assert report['validation_samples'] == model['validation_samples']
                    assert Path(report['predictions']).is_file()
                    continue
                no_other_job()
                attempt = int(previous.get('attempt', 0)) + 1
                destination = output / model['name'] / mode / f'attempt_{attempt:02d}'
                destination.parent.mkdir(parents=True, exist_ok=True)
                log = destination.parent / f'attempt_{attempt:02d}.log'
                tool = root / 'projects/myfusion/tools' / (
                    'evaluate_kitti_camera_case.py' if model['dataset'] == 'kitti'
                    else 'evaluate_nuscenes_camera_case.py')
                command = [sys.executable, '-u', str(tool), '--config', model['config'],
                    '--checkpoint', model['checkpoint'], '--mode', mode,
                    '--work-dir', str(destination)]
                if model['dataset'] == 'kitti':
                    command += ['--memory-fraction', '0.85']
                state['active_case'] = key
                case = dict(status='running', attempt=attempt, started_at=now(),
                            report=str(destination / 'metrics.json'), log=str(log))
                state['cases'][key] = case
                write(path, state)
                print('START', key, now(), flush=True)
                with log.open('x') as stream:
                    child = subprocess.Popen(command, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    case.update(evaluation_pid=child.pid, process_group=child.pid)
                    write(path, state)
                    result = child.wait()
                child = None
                case.update(completed_at=now(), returncode=result)
                if result:
                    case['status'] = 'failed'
                    print('FAILED', key, 'see', log, flush=True)
                else:
                    report = json.loads(Path(case['report']).read_text())
                    assert report['status'] == 'passed'
                    assert report['strict_checkpoint_load_passed']
                    assert report['checkpoint_sha256'] == model['checkpoint_sha256']
                    assert report['validation_samples'] == model['validation_samples']
                    case['status'] = 'passed'
                    case['metrics_percentage_points'] = metrics_for(model, report)
                    print('DONE', key, case['metrics_percentage_points'], flush=True)
                write(path, state)
                summarize(output, state)
        state.update(status=('complete' if all(c['status'] == 'passed'
            for c in state['cases'].values()) else 'complete_with_failures'),
            active_case=None, completed_at=now())
    except BaseException as error:
        if child is not None and child.poll() is None:
            import signal
            os.killpg(child.pid, signal.SIGINT)
            child.wait()
        state.update(status='interrupted', error=repr(error))
        raise
    finally:
        write(path, state)
    print('SUITE FINISHED', state['status'], flush=True)


if __name__ == '__main__':
    main()
