"""Publish compact verified reports, never datasets, checkpoints or predictions.

Input is the metrics-only snapshot produced by export_research_metrics.py.
All source bytes are checked against its manifest before copying. An existing
nonempty destination is refused. Only Python's standard library is required.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curate(snapshot, destination):
    snapshot, destination = Path(snapshot), Path(destination)
    original = json.loads((snapshot / 'MANIFEST.json').read_text())
    remote_root = Path(original['remote_root'])
    expected = {x['path']: x['sha256'] for x in original['files']}
    suite_rel = Path('work_dirs/camera_ablations_kitti_nuscenes_20261025')
    suite = json.loads((snapshot / suite_rel / 'suite.json').read_text())
    if (suite['status'] != 'complete' or len(suite['cases']) != 30 or
            not all(x['status'] == 'passed' for x in suite['cases'].values())):
        raise ValueError('Expected 30 completed full-validation cases')
    files = {}

    def include(source, output):
        source, output = Path(source), Path(output)
        path = snapshot / source
        if source.as_posix() not in expected or digest(path) != expected[source.as_posix()]:
            raise ValueError(f'Source digest mismatch: {source}')
        if path.stat().st_size > 1024 * 1024:
            raise ValueError(f'Not a compact report: {source}')
        files[output] = source

    for name in ['suite.json', 'comparison.json', 'SUMMARY.md', 'overall_metrics.csv']:
        include(suite_rel / name, Path('camera_ablations') / name)
    for model in suite['models']:
        name = model['name']
        config = Path(model['config']).relative_to(remote_root)
        include(config, Path('saved_configs') / (name + '.py'))
        if digest(snapshot / config) != model['config_sha256']:
            raise ValueError(f'Saved training config hash mismatch: {name}')
        for mode in ['correct', 'black', 'shuffled']:
            matches = [case for case in suite['cases'].values()
                       if f'/{name}/{mode}/' in case['report']]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one completed {name}/{mode} case')
            report = Path(matches[0]['report']).relative_to(remote_root)
            values = json.loads((snapshot / report).read_text())
            if (not values['strict_checkpoint_load_passed'] or
                    values['checkpoint_sha256'] != model['checkpoint_sha256'] or
                    values['validation_samples'] != model['validation_samples']):
                raise ValueError(f'Case identity mismatch: {name}/{mode}')
            include(report, Path('camera_ablations') / name / (mode + '.json'))
            if model['dataset'] == 'nuscenes':
                include(report.parent / 'results/pred_instances_3d/metrics_summary.json',
                        Path('camera_ablations') / name / (mode + '_devkit.json'))
    extras = {
        'baselines/kitti_pointpillars.json': 'kitti_camera_ablations_20261018_v2b8_vs_legacyb8/lidar/correct/metrics.json',
        'baselines/kitti_pointpillars_original.json': 'pointpillars_lidar_control_eb48_stable_momentum_seed0/best_epoch28_eval/20260816_235958/20260816_235958.json',
        'baselines/nuscenes_pointpillars.json': 'nuscenes_analysis/baseline/results/pred_instances_3d/metrics_summary.json',
        'baselines/nuscenes_transfusion_l.json': 'transfusion_l_pretrained_baseline/results/pred_instances_3d/metrics_summary.json',
        'baselines/transfusion_l_verification.json': 'transfusion_l_pretrained_baseline/baseline_verification.json',
        'historical/nuscenes_original_comparison.json': 'nuscenes_analysis/report/metrics_comparison.json',
        'historical/nuscenes_distance_stratified_map.json': 'nuscenes_analysis/report/distance_stratified_map.json',
        'saved_configs/kitti_original_paper.py': 'mbt_bev_camera_focused_eb48_seed0/20260818_190738/vis_data/config.py',
        'training/transfusion_l_integrity.json': 'transfusion_l_mbt_swint_e256_b8_seed0/transfusion_l_integrity.json',
        'training/transfusion_l_progress.json': 'transfusion_l_mbt_swint_e256_b8_seed0/training_progress.json',
    }
    for output, source in extras.items():
        include(Path('work_dirs') / source, output)
    # Preserve only validation entries, not thousands of training-loss rows.
    history_sources = {
        'kitti_original_paper': 'mbt_bev_camera_focused_eb48_seed0/20260818_190738/vis_data/scalars.json',
        'kitti_pointpillars_original': 'pointpillars_lidar_control_eb48_stable_momentum_seed0/20260815_212802/vis_data/scalars.json',
    }
    histories = {}
    for name, suffix in history_sources.items():
        source = Path('work_dirs') / suffix
        path = snapshot / source
        if digest(path) != expected[source.as_posix()]:
            raise ValueError(source)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [row for row in rows if any('Overall_3D_AP40_moderate' in k for k in row)]
        if not rows:
            raise ValueError(f'No completed validation scores: {name}')
        histories[name] = dict(source=str(source), source_sha256=digest(path), rows=rows)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Use a new output directory')
    destination.mkdir(parents=True, exist_ok=True)
    provenance = []
    for output, source in sorted(files.items()):
        target = destination / output
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / source, target)
        provenance.append(dict(path=str(output), source=str(source),
                               sha256=digest(target), bytes=target.stat().st_size))
    for name, history in histories.items():
        output = Path('historical') / (name + '_validation_history.json')
        target = destination / output
        target.write_text(json.dumps(history, indent=2) + '\n')
        provenance.append(dict(path=str(output), source=history['source'],
                               source_sha256=history['source_sha256'],
                               sha256=digest(target), bytes=target.stat().st_size,
                               transformation='Retain completed AP40 validation records only'))
    manifest = dict(snapshot_manifest_sha256=digest(snapshot / 'MANIFEST.json'),
                    source_remote_root=str(remote_root), completed_camera_cases=30,
                    exclusion='No model weights, datasets, raw predictions, logs or temporary files',
                    files=provenance)
    (destination / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = curate(args.snapshot, args.output)
    print(f'Published {len(result["files"])} verified compact artifacts to {args.output}')
