"""Controlled gated-ABFuse study definitions; no training on import."""
from copy import deepcopy


def study_arms(dataset):
    if dataset not in ('kitti', 'nuscenes'):
        raise ValueError(dataset)
    view = {'camera_view_drop_prob': .15} if dataset == 'nuscenes' else {}
    return [
        dict(name='average_no_dropout', changes=dict(use_bottleneck_gate=False,
            camera_drop_prob=0., **({'camera_view_drop_prob': 0.} if view else {}))),
        dict(name='average_dropout', changes=dict(use_bottleneck_gate=False)),
        dict(name='gate_no_dropout', changes=dict(camera_drop_prob=0.,
            **({'camera_view_drop_prob': 0.} if view else {}))),
        *[dict(name=f'b{b}', changes=dict(num_bottleneck_tokens=b)) for b in (1, 2, 4, 16)],
        dict(name='width128', changes=dict(embed_dim=128)),
        dict(name='width512', changes=dict(embed_dim=512)),
    ]


def changed_paths(first, second, prefix=''):
    """Leaf differences for a saved resolved config contract."""
    if isinstance(first, dict) and isinstance(second, dict):
        out = []
        for key in sorted(first.keys() | second.keys()):
            path = f'{prefix}.{key}' if prefix else str(key)
            if key not in first or key not in second:
                out.append(path)
            else:
                out += changed_paths(first[key], second[key], path)
        return out
    return [] if first == second else [prefix]


def apply_arm(base, arm):
    cfg = deepcopy(base)
    cfg['model']['fusion_module'].update(arm['changes'])
    assert changed_paths(base, cfg) == sorted(
        'model.fusion_module.' + k for k in arm['changes'])
    return cfg


def cold_start_reference(base, dataset):
    cfg=deepcopy(base)
    cfg['resume']=False
    if dataset=='nuscenes':
        # Verified initialization hooks load native LiDAR + nuImages weights.
        cfg['load_from']=None
    else:
        assert cfg.get('load_from'), 'KITTI requires its LiDAR-only initializer'
    return cfg


def aligned_kitti_checkpoints(run):
    """Prefer an archived full checkpoint at an accumulation-aligned epoch."""
    import re
    candidates=[]
    for p in run.glob('*.pth'):
        match=re.fullmatch(r'(aligned_)?epoch_(\d+)\.pth',p.name)
        if match and int(match[2])%4==0 and p.is_file():
            candidates.append((int(match[2]),bool(match[1]),p))
    return [x[2] for x in sorted(candidates)]
