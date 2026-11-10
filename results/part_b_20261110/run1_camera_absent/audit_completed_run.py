"""Read-only extraction and final checkpoint audit; emits compact JSON only."""
from pathlib import Path
import hashlib
import json
import re
import torch

BASE = Path('/home/yab/mmdetection3d')
RUN = BASE / 'work_dirs/part_b_kitti_absent_full_20261109'
ATTEMPT = BASE / 'work_dirs/part_b_run1_resume_20261110'
files = {}
for name, path in {
    'resumed_config.py': RUN / 'absent_resolved_config.py',
    'training_progress.json': RUN / 'absent_training_progress.json',
    'matched_recipe_verification.json': RUN / 'matched_recipe_verification.json',
    'resume_report.json': ATTEMPT / 'resume_report.json',
    'launch_run1.py': BASE / 'work_dirs/start_part_b_kitti_absent_full_20261109.py',
    'resume_run1.py': ATTEMPT / 'resume_run1.py',
}.items():
    files[name] = path.read_text()
progress = json.loads(files['training_progress.json'])
assert progress['status'] == 'complete'
assert progress['completed_epochs'] == 40 and progress['optimizer_updates'] == 6190
logs = [BASE / 'work_dirs/part_b_kitti_absent_full_20261109.log', ATTEMPT / 'training.log']
history, raw_records = [], {}
for log in logs:
    for line in log.read_text().splitlines():
        if 'Epoch(val)' not in line or 'Overall_3D_AP40_moderate:' not in line:
            continue
        epoch = int(re.search(r'Epoch\(val\) \[(\d+)\]', line).group(1))
        def metric(kind):
            return float(re.search(r'Overall_' + kind + r'_AP40_moderate: ([0-9.]+)', line).group(1))
        history.append(dict(epoch=epoch, moderate_3d_ap40=metric('3D'), moderate_bev_ap40=metric('BEV')))
        assert epoch not in raw_records
        raw_records[epoch] = line
assert [h['epoch'] for h in history] == list(range(2, 41, 2))
selected = max(history, key=lambda h: h['moderate_3d_ap40'])
checkpoint, = RUN.glob('best*.pth')
saved = torch.load(checkpoint, map_location='cpu')
assert saved['meta']['epoch'] == selected['epoch'] == 14
baseline, = (BASE / 'work_dirs/pointpillars_lidar_control_eb48_stable_momentum_seed0').glob('best*epoch_28.pth')
reference = torch.load(baseline, map_location='cpu')['state_dict']
from projects.myfusion.fusion.mbt_detector_camera_v2 import CameraFocusedMBTBEVFusionDetectorV2
CameraFocusedMBTBEVFusionDetectorV2._remap_pointpillars_state_dict(reference)
prefixes = ('voxel_encoder.', 'middle_encoder.', 'backbone.', 'neck.', 'bbox_head.')
protected = {n: t for n, t in saved['state_dict'].items() if n.startswith(prefixes)}
assert len(protected) == 126
for name, value in protected.items():
    assert torch.equal(value, reference[name]), name
from mmengine.config import Config
cfg = Config.fromfile(RUN / 'absent_resolved_config.py').to_dict()
gated_path = BASE / 'work_dirs/part_b_kitti_gated_full_20261109/gated_resolved_config.py'
gated = Config.fromfile(gated_path).to_dict()
assert cfg['resume'] is True
assert cfg['load_from'] == str(RUN / 'epoch_12.pth')
assert gated['resume'] is False
baseline_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
assert baseline_hash == progress['baseline_sha256']
for config in (cfg, gated):
    # Runner re-dumps the resumed config using the original filename. Preserve
    # that effective resumed config separately from the pre-resume local copy.
    config.pop('resume')
    config.pop('load_from')
    for name in ('camera_mode', 'bypass_missing'):
        config['model']['fusion_module'].pop(name)
    config.pop('work_dir')
    for mode in ('val', 'test'):
        config[mode + '_evaluator'].pop('pklfile_prefix')
assert cfg == gated
files['selected_validation_record.txt'] = raw_records[selected['epoch']] + '\n'
files['validation_history.json'] = json.dumps(history, indent=2) + '\n'
summary = dict(status='complete', dataset='KITTI validation, all 3769 frames',
               metric='Overall moderate AP40, mean across Car/Pedestrian/Cyclist',
               selection='best moderate 3D AP40; BEV at the same selected checkpoint',
               selected_checkpoint=str(checkpoint),
               checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               resumed_config_sha256=hashlib.sha256((RUN / 'absent_resolved_config.py').read_bytes()).hexdigest(),
               baseline_sha256=baseline_hash,
               selected_metrics=selected,
               protected_lidar_tensors_exactly_verified=len(protected),
               resolved_recipe_comparison_rechecked=True,
               config_record_note='MMEngine rewrote absent_resolved_config.py with resume=True and epoch_12 load_from during recovery. This effective config is exported as resumed_config.py; the original local pre-resume copy is exported separately as initial_config.py.',
               validation_log_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in logs},
               completion=progress,
               caveat='Single seed, interrupted training; no bitwise RNG-continuity claim. Missing camera supervision and retained residual gradient routing differ from the full-camera arm. Extra-capacity control, not perfect causal isolation.')
files['summary.json'] = json.dumps(summary, indent=2) + '\n'
print(json.dumps(dict(files=files, source_root=str(RUN), audited_at=__import__('datetime').datetime.now().astimezone().isoformat())))
