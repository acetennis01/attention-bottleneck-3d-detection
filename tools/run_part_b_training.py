"""Run two bounded KITTI pilots, then fresh gated TransFusion/Swin training.

Requires passed real-data preflights. Runs sequentially on one GPU; never
resumes smoke weights or overwrites an existing experiment directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mmengine.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-epochs', type=int, default=2)
    parser.add_argument('--tag', default='part_b_20261107')
    args = parser.parse_args()
    if args.pilot_epochs < 1 or args.pilot_epochs > 4:
        parser.error('Bounded KITTI pilot must use 1 to 4 epochs')
    if not args.tag.replace('_', '').isalnum():
        parser.error('tag must be alphanumeric with underscores')
    root = Path('work_dirs', args.tag + '_training_queue').resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    state = dict(status='preflight_check', pilot_epochs=args.pilot_epochs,
                 stages=[], source_root=str(Path.cwd()))

    def write():
        temp = root / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(root / 'status.json')

    write()
    try:
        for name in ('kitti', 'transfusion'):
            path = Path('work_dirs', args.tag + '_' + name + '_preflight',
                        'report.json')
            report = json.loads(path.read_text())
            assert report['status'] == 'passed', path
        plans = [
            ('kitti_absent_pilot', 'kitti_gated_absent_control.py'),
            ('kitti_gated_pilot', 'kitti_gated_b8.py'),
            ('transfusion_gated_full', 'transfusion_l_gated_swint_nuscenes.py')]
        configs = []
        for stage, filename in plans:
            cfg = Config.fromfile('projects/myfusion/configs/' + filename)
            destination = Path('work_dirs', args.tag + '_' + stage).resolve()
            if destination.exists():
                raise FileExistsError(destination)
            cfg.work_dir = str(destination)
            cfg.resume = False
            if stage.startswith('kitti'):
                # Retain the planned 40-epoch LR schedule; this is an early
                # truncated pilot, not a new short cosine or final comparison.
                cfg.train_cfg.max_epochs = args.pilot_epochs
                cfg.train_cfg.val_interval = 1
                cfg.val_evaluator.pklfile_prefix = str(destination / 'val_predictions')
                cfg.test_evaluator.pklfile_prefix = str(destination / 'test_predictions')
            else:
                cfg.val_evaluator.jsonfile_prefix = str(destination / 'validation')
                cfg.test_evaluator.jsonfile_prefix = str(destination / 'validation')
            path = root / (stage + '.py')
            cfg.dump(path)
            configs.append((stage, path, destination))
            state['stages'].append(dict(name=stage, status='pending',
                work_dir=str(destination), config=str(path),
                config_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        state['status'] = 'running'
        write()
        for entry, (stage, config, destination) in zip(state['stages'], configs):
            entry['status'] = 'running'
            state['current_stage'] = stage
            write()
            with (root / (stage + '.log')).open('x') as log:
                subprocess.run([sys.executable,
                    '/home/yab/mmdetection3d/tools/train.py', str(config)],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            entry['status'] = 'complete'
            write()
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        write()


if __name__ == '__main__':
    main()
