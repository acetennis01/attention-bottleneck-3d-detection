"""GPU smoke: public-weight loading, baseline equivalence and MBT gradients.

Small smoke updates are discarded, never used as a training initialization.
Run from MMDetection3D root. Does not mark full baseline reproduction passed.
"""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import set_random_seed
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules

from projects.myfusion.tools.transfusion_l_checkpoint import load_lidar_checkpoint


def same_tree(first, second, path='output', exact=True):
    if isinstance(first, torch.Tensor):
        if not exact:
            torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-3)
        elif not torch.equal(first, second):
            difference = (first - second).abs()
            raise AssertionError(f'{path}: max_abs={difference.max().item()}, '
                f'first_finite={torch.isfinite(first).all().item()}, '
                f'second_finite={torch.isfinite(second).all().item()}, '
                f'first_range=({first.min().item()},{first.max().item()}), '
                f'shapes={first.shape}/{second.shape}')
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            same_tree(first[key], second[key], path + '.' + key, exact)
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for index, (a, b) in enumerate(zip(first, second)):
            same_tree(a, b, path + f'[{index}]', exact)
    else:
        assert first == second


def max_tree_difference(first, second):
    if isinstance(first, torch.Tensor):
        return float((first - second).abs().max()) if first.numel() else 0.
    if isinstance(first, dict):
        return max((max_tree_difference(first[key], second[key]) for key in first), default=0.)
    if isinstance(first, (list, tuple)):
        return max((max_tree_difference(a, b) for a, b in zip(first, second)), default=0.)
    return 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='work_dirs/transfusion_l_mbt_smoke/report.json')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    set_random_seed(0)
    register_all_modules(init_default_scope=True)
    base = Config.fromfile('projects/myfusion/configs/transfusion_l_nuscenes_pretrained.py')
    cfg = Config.fromfile('projects/myfusion/configs/transfusion_l_mbt_nuscenes.py')
    baseline = MODELS.build(deepcopy(base.model)).cuda().eval()
    base_transfer = load_lidar_checkpoint(baseline, base.lidar_checkpoint)
    model = MODELS.build(deepcopy(cfg.model)).cuda().eval()
    transfer = load_lidar_checkpoint(model, cfg.lidar_checkpoint, fusion=True)

    # Real CUDA voxelizer coordinates, not a square-shape assumption.
    toy = torch.tensor([[-53.8875, -53.6625, -4.7, .5, 0]], device='cuda')
    _, coordinates, _ = baseline.pts_voxel_layer(toy)
    torch.testing.assert_close(coordinates.cpu(), torch.tensor([[1, 4, 1]], dtype=torch.int32))
    dataset = DATASETS.build(deepcopy(base.test_dataloader.dataset))
    assert len(dataset) == 6019
    baseline_example = dataset[0]
    del dataset
    dataset = DATASETS.build(deepcopy(cfg.test_dataloader.dataset))
    assert len(dataset) == 6019
    fusion_example = dataset[0]
    del dataset
    assert torch.equal(baseline_example['inputs']['points'], fusion_example['inputs']['points'])
    baseline_batch = baseline.data_preprocessor(pseudo_collate([baseline_example]), False)
    batch = model.data_preprocessor(pseudo_collate([fusion_example]), False)
    metas = [sample.metainfo for sample in batch['data_samples']]
    with torch.no_grad():
        original = baseline.extract_feat(baseline_batch['inputs'],
            [sample.metainfo for sample in baseline_batch['data_samples']])
        fused = model.extract_feat(batch['inputs'], metas)
        same_tree(original, fused)
        assert torch.isfinite(fused[0]).all()
        raw_original = baseline.bbox_head(original, metas)
        raw_fused = model.bbox_head(fused, metas)
        assert all(torch.equal(value, model.bbox_head.state_dict()[key])
                   for key, value in baseline.bbox_head.state_dict().items())
        print('HEAD CHECK', baseline.bbox_head.training, model.bbox_head.training,
              original[0].stride(), fused[0].stride(), flush=True)
        # GPU head execution may differ slightly despite bitwise-identical
        # input tensors and all head parameters. Do not label this bitwise.
        raw_repeat = baseline.bbox_head(original, metas)
        raw_head_difference = max_tree_difference(raw_original, raw_fused)
        repeated_head_difference = max_tree_difference(raw_original, raw_repeat)
        print('HEAD NUMERICS', raw_head_difference, repeated_head_difference, flush=True)
        same_tree(raw_original, raw_fused, exact=False)
        predictions = model.predict(batch['inputs'], batch['data_samples'])
        instances = predictions[0].pred_instances_3d
        assert torch.isfinite(instances.bboxes_3d.tensor).all()
        assert torch.isfinite(instances.scores_3d).all()
        count = len(instances.scores_3d)
    protected = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                 if key.startswith(('pts_', 'bbox_head.'))}
    del baseline, original, fused, baseline_batch, raw_original, raw_fused
    gc.collect()
    torch.cuda.empty_cache()

    train_data = DATASETS.build(deepcopy(cfg.train_dataloader.dataset))
    train_batch = model.data_preprocessor(pseudo_collate([train_data[0]]), True)
    del train_data
    model.train()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                  if parameter.requires_grad), lr=1e-4)
    steps = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(train_batch['inputs'], train_batch['data_samples'])
        total, _ = model.parse_losses(losses)
        assert torch.isfinite(total)
        total.backward()
        groups = dict(camera=model.img_backbone, image_fpn=model.img_neck,
                      fusion=model.fusion_module)
        gradients = {name: sum(float(parameter.grad.detach().float().square().sum())
            for parameter in module.parameters() if parameter.grad is not None) ** .5
            for name, module in groups.items()}
        bottleneck = model.fusion_module.bottleneck_tokens.grad
        gradients['bottleneck'] = 0. if bottleneck is None else float(bottleneck.norm())
        assert gradients['camera'] > 0 and gradients['image_fpn'] > 0
        if step == 1:
            assert gradients['bottleneck'] > 0
        for name in model.lidar_module_names:
            module = getattr(model, name)
            assert not module.training
            assert all(parameter.grad is None for parameter in module.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        steps.append(dict(loss=float(total.detach()), gradients=gradients))
    state = model.state_dict()
    assert all(torch.equal(state[key].cpu(), value) for key, value in protected.items())
    torch.cuda.synchronize()
    report = dict(status='passed', strict_baseline_transfer=base_transfer,
        fusion_transfer=transfer, validation_frames=6019,
        identical_point_inputs=True, voxel_coordinates_order='x,y,z',
        fusion_transposes_xy_to_yx_and_back=True,
        zero_residual_bev_features_bitwise_identical=True,
        pretrained_head_parameters_bitwise_identical=True,
        head_outputs_numerically_equivalent=True,
        head_comparison_rtol=1e-4, head_comparison_atol=1e-3,
        head_max_abs_difference=raw_head_difference,
        repeated_baseline_head_max_abs_difference=repeated_head_difference,
        finite_predictions=count, smoke_optimizer_steps=steps,
        frozen_lidar_and_head_unchanged=True,
        peak_cuda_allocated_GiB=torch.cuda.max_memory_allocated() / 1024 ** 3,
        full_validation_reproduction_passed=False,
        note='Smoke-updated weights discarded; full baseline evaluation is still required.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
