"""Create a checksummed metrics-only archive; never include weights/predictions."""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def add_text(archive, name, value):
    payload = value.encode()
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    suite_path = Path(args.suite).resolve()
    suite = json.loads((suite_path / 'suite.json').read_text())
    assert suite['status'] == 'complete'
    assert len(suite['cases']) == suite['total_cases'] == 30
    assert all(case['status'] == 'passed' for case in suite['cases'].values())
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    trees = [suite_path]
    trees += [Path(model['config']).parent for model in suite['models']]
    extra = [
        'transfusion_l_pretrained_baseline',
        'transfusion_l_mbt_swint_initialization',
        'transfusion_l_swint_runner_smoke_20261022b',
        'pointpillars_lidar_control_eb48_stable_momentum_seed0',
        'pointpillars_nuscenes',
        'mbt_bev_camera_focused_eb48_seed0',
        'kitti_camera_ablations_20261018_v2b8_vs_legacyb8',
        'nuscenes_analysis/baseline', 'nuscenes_analysis/correct',
        'nuscenes_analysis/black', 'nuscenes_analysis/shuffled',
        'nuscenes_analysis/report',
    ]
    trees += [root / 'work_dirs' / name for name in extra]
    files = set()
    missing = []
    allowed = {'.json', '.csv', '.md', '.log', '.txt', '.png', '.pdf'}
    forbidden = {'results_nusc.json', 'last_checkpoint'}
    for tree in trees:
        if not tree.exists():
            missing.append(str(tree))
            continue
        for path in tree.rglob('*'):
            if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts:
                continue
            is_config = path.suffix == '.py' and (
                path.name == 'config.py' or path.parent == tree)
            if (path.suffix in allowed or is_config) and path.name not in forbidden:
                files.add(path)
    for name in ('transfusion_l_mbt_swint_e256_b8_seed0_train.log',
                 'transfusion_l_pretrained_baseline_launch.log',
                 suite_path.name + '_queue.log'):
        path = root / 'work_dirs' / name
        if path.is_file():
            files.add(path)
    entries = []
    for path in sorted(files):
        relative = path.relative_to(root)
        if path.stat().st_size > 100 * 1024 * 1024:
            raise RuntimeError(f'Unexpectedly large metric artifact: {path}')
        entries.append(dict(path=str(relative), bytes=path.stat().st_size,
                            sha256=digest(path)))
    manifest = dict(created_at=datetime.now().astimezone().isoformat(),
        remote_root=str(root), completed_camera_ablation_cases=30,
        exported_files=len(entries), uncompressed_bytes=sum(e['bytes'] for e in entries),
        excluded=['model weights (*.pth)', 'raw predictions (*.pkl, results_nusc.json)',
                  'datasets', 'videos', 'compiler/runtime files'],
        missing_optional_directories=missing, files=entries)
    label = 'work_dirs/' + suite_path.name
    readme = f'''# KITTI and nuScenes research metrics

Downloaded snapshot of all 30 completed camera-ablation evaluations, associated
model training histories, validation metrics, baseline reports and saved configs.
Source: Haddr's PC (`{root}`). Scores are validation results, not test-server scores.

## Start here

- [All camera results (table)]({label}/SUMMARY.md)
- [All camera results (CSV)]({label}/overall_metrics.csv)
- [Camera drops and exact scores (JSON)]({label}/comparison.json)
- [Selected checkpoints, config hashes, all case statuses]({label}/suite.json)

Every model has correct/black/shuffled evaluations using the SAME selected
checkpoint: KITTI B1/B2/B4/B8/B16, larger KITTI V2 B8/B16, original nuScenes MBT,
multi-scale nuScenes MBT and TransFusion-L/Swin MBT. KITTI uses all 3,769 validation
frames; nuScenes uses all 6,019. Checkpoint filenames and SHA-256 hashes are saved.

Per-case `metrics.json` files contain KITTI class/difficulty metrics and verification
metadata. nuScenes `results/pred_instances_3d/metrics_summary.json` files include
per-class AP and NDS components; `metrics_details.json` includes metric curves.
Black/shuffled drops support camera dependence under corruption; they are NOT
proof of statistical significance or attribution of the entire LiDAR-baseline gain.

## Training and baselines

Corresponding `work_dirs/<training-run>/` directories include resolved configs,
training/scalar histories and available checkpoint/initialization integrity reports.
nuScenes `validation/iter_*/` preserves the completed validation history. The
TransFusion-L/Swin run completed 160,000 batches / 40,000 optimizer updates.
The interrupted validation at iteration 150,000 has no completed score; use the
completed 160,000 result or selected best-NDS checkpoint at iteration 110,000.
Best mAP (80,000) and best NDS (110,000) are DIFFERENT checkpoint selections.

- [Strong LiDAR baseline verification](work_dirs/transfusion_l_pretrained_baseline/baseline_verification.json)
- [New-model final integrity](work_dirs/transfusion_l_mbt_swint_e256_b8_seed0/transfusion_l_integrity.json)
- [Earlier nuScenes analysis](work_dirs/nuscenes_analysis/report/README.md)

The earlier `nuscenes_analysis` and 2026-10-18 KITTI reports refer to historical
models. Do not confuse their numbers with the new TransFusion-L/Swin experiment.
No existing FLOPs/latency profiling report was found in the selected directories;
none is invented by this export.

## What is not included

Large model weights, raw per-frame predictions, datasets and videos remain on
Haddr's PC. Their source paths are retained in the reports. All included files
have byte counts and SHA-256 checksums in [MANIFEST.json](MANIFEST.json).
'''
    with tarfile.open(output, 'w:gz') as archive:
        for entry in entries:
            archive.add(root / entry['path'], arcname=entry['path'], recursive=False)
        add_text(archive, 'MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        add_text(archive, 'README.md', readme)
    print(json.dumps(dict(archive=str(output), archive_bytes=output.stat().st_size,
                         archive_sha256=digest(output), exported_files=len(entries),
                         uncompressed_bytes=manifest['uncompressed_bytes'],
                         missing_optional_directories=missing), indent=2), flush=True)


if __name__ == '__main__':
    main()
