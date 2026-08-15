"""Audit MBT configuration, optimizer coverage, and checkpoint tensors."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoints', nargs='*')
    return parser.parse_args()


def component(name: str) -> str:
    for prefix, group in (
            ('img_backbone.', 'image_backbone'),
            ('fusion_module.', 'fusion'),
            ('voxel_encoder.', 'lidar_encoder'),
            ('middle_encoder.', 'lidar_encoder'),
            ('backbone.', 'lidar_encoder'),
            ('neck.', 'lidar_encoder'),
            ('bbox_head.', 'bbox_head')):
        if name.startswith(prefix):
            return group
    return 'other'


def audit_optimizer(cfg: Config) -> dict:
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    import_modules_from_strings(**cfg.custom_imports)
    torch.manual_seed(cfg.get('randomness', {}).get('seed', 0))
    model = MODELS.build(cfg.model)
    model.init_weights()
    wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
    all_named = {id(p): name for name, p in model.named_parameters()}
    trainable = {id(p): name for name, p in model.named_parameters()
                 if p.requires_grad}
    optimized = {
        id(p) for group in wrapper.optimizer.param_groups
        for p in group['params']
    }
    missing = sorted(name for pid, name in trainable.items()
                     if pid not in optimized)
    frozen_optimized = sorted(
        all_named.get(pid, f'<unknown:{pid}>')
        for pid in optimized - set(trainable))
    counts = defaultdict(int)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            counts[component(name)] += parameter.numel()

    print('Optimizer coverage')
    for name, count in sorted(counts.items()):
        print(f'  {name}: {count:,} trainable parameters')
    print(f'  missing trainable tensors: {len(missing)}')
    for name in missing:
        print(f'    {name}')
    print(f'  frozen tensors retained by optimizer: {len(frozen_optimized)}')
    for name in frozen_optimized:
        print(f'    {name}')
    print('  learning rates:',
          sorted({group['lr'] for group in wrapper.optimizer.param_groups}))
    batch_size = cfg.train_dataloader.batch_size
    accumulation = cfg.optim_wrapper.get('accumulative_counts', 1)
    effective_batch = batch_size * accumulation
    expected_batch = cfg.get('auto_scale_lr', {}).get('base_batch_size')
    print(f'  effective batch per optimizer update: {effective_batch} '
          f'(batch={batch_size}, accumulation={accumulation})')
    if expected_batch is not None:
        print(f'  schedule reference batch size: {expected_batch}')
        if effective_batch != expected_batch:
            print('  WARNING: effective batch does not match the schedule '
                  'reference batch size')
    if missing:
        raise RuntimeError('optimizer coverage audit failed')
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if torch.is_tensor(tensor)
    }


def state_dict_from(path: Path):
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint.get('state_dict', checkpoint)


def audit_checkpoint(path: Path):
    state = state_dict_from(path)
    totals = defaultdict(lambda: [0, 0.0])
    nonfinite = []
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        value = tensor.detach().float()
        if not torch.isfinite(value).all():
            nonfinite.append(name)
        group = component(name)
        totals[group][0] += tensor.numel()
        totals[group][1] += value.square().sum().item()

    print(f'Checkpoint: {path}')
    for name, (count, squared) in sorted(totals.items()):
        print(f'  {name}: tensors={count:,} l2={math.sqrt(squared):.6g}')
    for key in (
            'fusion_module.bottleneck_tokens',
            'fusion_module.image_projection.weight',
            'fusion_module.lidar_projection.weight',
            'fusion_module.bev_residual.weight'):
        if key in state:
            print(f'  {key}: l2={state[key].float().norm().item():.6g}')
    print(f'  non-finite tensors: {len(nonfinite)}')
    for name in nonfinite:
        print(f'    {name}')
    if nonfinite:
        raise RuntimeError(f'non-finite tensors in {path}')
    return state


def compare(first_path: Path, first, second_path: Path, second) -> None:
    squared = defaultdict(float)
    counts = defaultdict(int)
    common = sorted(set(first) & set(second))
    for name in common:
        if not torch.is_tensor(first[name]) or first[name].shape != second[name].shape:
            continue
        delta = first[name].float() - second[name].float()
        group = component(name)
        squared[group] += delta.square().sum().item()
        counts[group] += delta.numel()
    print(f'Checkpoint delta: {first_path.name} -> {second_path.name}')
    for name in sorted(squared):
        rms = math.sqrt(squared[name] / max(counts[name], 1))
        print(f'  {name}: parameter_delta_rms={rms:.6g}')


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    initialized = audit_optimizer(cfg)
    audited = []
    for raw_path in args.checkpoints:
        path = Path(raw_path)
        audited.append((path, audit_checkpoint(path)))
    if audited:
        compare(Path('initialized_model'), initialized,
                audited[0][0], audited[0][1])
    if len(audited) == 2:
        compare(audited[0][0], audited[0][1],
                audited[1][0], audited[1][1])


if __name__ == '__main__':
    main()
