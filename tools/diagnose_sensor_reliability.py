"""Read-only detector probes and disposable quality-head refits on cached data.

Uses train frames for diagnostic head fitting and separate validation frames
for assessment. No detector checkpoint is updated or selected. Refitted heads
are discarded. These are severity-regression diagnostics, not new detection AP.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.structures.points import LiDARPoints
from mmdet3d.utils import register_all_modules

from projects.myfusion.tools.run_gated_studies import sha, write
from projects.myfusion.transforms.sensor_reliability import corrupt_sensor_inputs


CASES = [('clean', 0.0)] + [
    (condition, severity)
    for condition in ('camera_dim', 'camera_blur', 'lidar_dropout', 'lidar_noise')
    for severity in (.25, .5, .75)]
SEVERITY_WEIGHTS = (.31640625, .3671875, .31640625)
CASE_WEIGHTS = torch.tensor([.25] + [.1875 * w for _ in range(4) for w in SEVERITY_WEIGHTS])
FROZEN = ('img_backbone', 'img_neck', 'voxel_encoder', 'middle_encoder',
          'backbone', 'neck', 'bbox_head')


def case_batch(raw, condition, severity):
    image = raw['inputs']['img'].permute(1, 2, 0).numpy()
    points = LiDARPoints(raw['inputs']['points'], points_dim=4)
    index = int(raw['data_samples'].metainfo['sample_idx'])
    image, points, target = corrupt_sensor_inputs(
        image, points, condition, severity, np.random.default_rng(index))
    result = deepcopy(raw)
    result['inputs']['img'] = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    result['inputs']['points'] = points.tensor
    result['data_samples'].set_metainfo(dict(modality_quality_target=target))
    return result, target


def spatial_statistics(features):
    flat = features.flatten(2)
    return torch.cat((flat.mean(-1), flat.var(-1, unbiased=False).add(1e-6).sqrt()), -1)


def collect(model, cfg, frames):
    datasets = {}
    train_cfg = deepcopy(cfg.train_dataloader.dataset.dataset)
    train_cfg.pipeline = [step for step in train_cfg.pipeline
                          if step.type != 'SensorReliabilityCorruption']
    datasets['train'] = DATASETS.build(train_cfg)
    datasets['validation'] = DATASETS.build(cfg.test_dataloader.dataset)
    assert len(datasets['train']) == 3712 and len(datasets['validation']) == 3769
    fusion = model.fusion_module
    cache = {}
    for split, dataset in datasets.items():
        selected = np.random.default_rng(20261120).choice(len(dataset), frames, replace=False)
        rows = {name: [] for name in ('camera', 'lidar', 'raw_camera', 'raw_lidar',
                                     'lidar_density', 'target', 'case', 'frame')}
        identities = []
        for number, index in enumerate(selected):
            raw = dataset[int(index)]
            identities.append(str(raw['data_samples'].metainfo['lidar_path']))
            for case, (condition, severity) in enumerate(CASES):
                sample, target = case_batch(raw, condition, severity)
                points = sample['inputs']['points']
                processed = model.data_preprocessor(pseudo_collate([sample]), training=False)
                inputs = processed['inputs']
                with torch.no_grad():
                    camera = model.extract_img_feat(inputs['imgs'])
                    lidar = model.extract_pts_feat(inputs['voxels'], 1)
                    metadata = model._projection_metadata(processed['data_samples'], inputs['imgs'])
                    aligned, valid, _ = fusion._aligned_image_tokens(
                        camera, metadata['lidar2img'], metadata['image_shapes'],
                        metadata['scale_factors'], metadata['padded_image_shape'],
                        return_sampling_weights=True)
                    aligned = fusion.aligned_camera_norm(aligned)
                    tokens = fusion._tokenize(lidar, fusion.lidar_projection)
                    camera_stats = fusion._statistics(aligned, valid)
                    lidar_stats = fusion._statistics(tokens, torch.ones_like(valid))
                    voxel_counts = inputs['voxels']['num_points'].float()
                    density = [math.log1p(len(points)), math.log1p(len(voxel_counts)),
                               voxel_counts.mean().item(),
                               voxel_counts.std(unbiased=False).item(),
                               float(len(voxel_counts)) / max(len(points), 1)]
                    for name, tensor in (('camera', camera_stats), ('lidar', lidar_stats),
                                         ('raw_camera', spatial_statistics(camera)),
                                         ('raw_lidar', spatial_statistics(lidar))):
                        rows[name].append(tensor.squeeze(0).cpu())
                rows['lidar_density'].append(torch.tensor(density))
                rows['target'].append(torch.tensor(target))
                rows['case'].append(case)
                rows['frame'].append(int(index))
            if (number + 1) % 8 == 0:
                print('cached', split, number + 1, '/', frames, flush=True)
        cache[split] = {name: torch.stack(values) if isinstance(values[0], torch.Tensor)
                        else torch.tensor(values) for name, values in rows.items()}
        cache[split]['sample_paths'] = identities
    assert not set(cache['train']['sample_paths']) & set(cache['validation']['sample_paths'])
    return cache


def assess(prediction, data, sensor):
    target = data['target'][:, sensor]
    weights = CASE_WEIGHTS[data['case']]
    weights = weights / weights.sum()
    rows = []
    for i, (condition, severity) in enumerate(CASES):
        mask = data['case'] == i
        rows.append(dict(condition=condition, severity=severity,
                         mean_prediction=float(prediction[mask].mean()),
                         target=float(target[mask][0]),
                         mae=float((prediction[mask] - target[mask]).abs().mean())))
    return dict(natural_weighted_mae=float(((prediction - target).abs() * weights).sum()),
                mean_sigmoid_derivative=float((prediction * (1 - prediction)).mean()),
                cases=rows)


def refit(initial, fit, test, feature_name, sensor, objective, steps=1200,
          learning_rate=1e-3):
    x, y = fit[feature_name], fit['target'][:, sensor]
    weights = CASE_WEIGHTS[fit['case']].clone()
    if objective == 'l1_corruption_weighted':
        corrupt = y < 1
        weights[corrupt] *= .75 / weights[corrupt].sum()
        weights[~corrupt] *= .25 / weights[~corrupt].sum()
    weights /= weights.sum()
    torch.manual_seed(0)
    head = deepcopy(initial)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=.01)
    generator = torch.Generator().manual_seed(0)
    for _ in range(steps):
        ids = torch.multinomial(weights, 128, replacement=True, generator=generator)
        logits = head(x[ids]).squeeze(-1)
        if objective == 'bce':
            loss = F.binary_cross_entropy_with_logits(logits, y[ids])
        else:
            loss = F.l1_loss(logits.sigmoid(), y[ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = head(test[feature_name]).squeeze(-1).sigmoid()
        fit_prediction = head(x).squeeze(-1).sigmoid()
    return dict(objective=objective, features=feature_name, steps=steps,
                learning_rate=learning_rate,
                train=assess(fit_prediction, fit, sensor),
                validation=assess(prediction, test, sensor))


def fresh_head(width):
    torch.manual_seed(0)
    head = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 1))
    nn.init.zeros_(head[-1].weight)
    nn.init.constant_(head[-1].bias, math.log(9))
    return head


def gradient_probes(model, cfg):
    dataset_cfg = deepcopy(cfg.train_dataloader.dataset.dataset)
    dataset_cfg.pipeline = [step for step in dataset_cfg.pipeline
                            if step.type != 'SensorReliabilityCorruption']
    dataset = DATASETS.build(dataset_cfg)
    rows = []
    heads = (model.fusion_module.camera_reliability, model.fusion_module.lidar_reliability)
    for condition, severity in [('clean', 0), ('camera_dim', .75), ('camera_blur', .75),
                                 ('lidar_dropout', .75), ('lidar_noise', .75)]:
        sample, target = case_batch(dataset[0], condition, severity)
        processed = model.data_preprocessor(pseudo_collate([sample]), training=False)
        model.zero_grad(set_to_none=True)
        losses = model(**processed, mode='loss')
        detection, _ = model.parse_losses({k: v for k, v in losses.items()
                                           if k not in ('loss_reliability', 'loss_camera_aux')})
        row = dict(condition=condition, severity=severity, target=target,
                   detection_loss=float(detection.detach()),
                   weighted_quality_loss=float(losses['loss_reliability'].detach()), sensors=[])
        for head in heads:
            parameters = list(head.parameters())
            det_grad = torch.autograd.grad(detection, parameters, retain_graph=True, allow_unused=True)
            quality_grad = torch.autograd.grad(losses['loss_reliability'], parameters, retain_graph=True)
            det = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                              for g, p in zip(det_grad, parameters)])
            quality = torch.cat([g.flatten() for g in quality_grad])
            assert torch.isfinite(det).all() and torch.isfinite(quality).all()
            denominator = det.norm() * quality.norm()
            row['sensors'].append(dict(detection_gradient_norm=float(det.norm()),
                                       quality_gradient_norm=float(quality.norm()),
                                       gradient_cosine=float(det.dot(quality) / denominator)
                                       if denominator > 0 else None,
                                       quality_final_bias_gradient=float(quality_grad[-1])))
        rows.append(row)
    model.zero_grad(set_to_none=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames-per-split', type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh diagnostic output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.4)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.study / 'guided.py')
    import_modules_from_strings(**cfg.custom_imports)
    checkpoint = args.study / 'guided/epoch_6.pth'
    completed = json.loads((args.study / 'guided/progress.json').read_text())
    assert completed['status'] == 'complete' and sha(checkpoint) == completed['checkpoint_sha256']
    model = MODELS.build(cfg.model).cuda().eval()
    weights = torch.load(checkpoint, map_location='cpu')['state_dict']
    model.load_state_dict(weights, strict=True)
    for name in FROZEN:
        getattr(model, name).requires_grad_(False).eval()
    report = dict(status='running', checkpoint_sha256=completed['checkpoint_sha256'],
                  frames_per_split=args.frames_per_split, steps_per_refit=1200,
                  note='Diagnostic head refits use training frames. Assessment uses separate validation frames. Detector weights are unchanged. No AP is measured.',
                  analytical_clean_target_probability=.625,
                  constant_l1_minimizer=1.0, constant_bce_minimizer=.8125)
    write(args.output / 'diagnosis.json', report)
    try:
        cache = collect(model, cfg, args.frames_per_split)
        torch.save(cache, args.output / 'feature_cache.pt')
        report['gradient_probes'] = gradient_probes(model, cfg)
        initializer = torch.load(args.study / 'initializer.pth', map_location='cpu')['state_dict']
        report['head_weight_changes'] = {}
        report['head_refits'] = {}
        fit, test = cache['train'], cache['validation']
        for sensor, name in enumerate(('camera', 'lidar')):
            head = deepcopy(getattr(model.fusion_module, name + '_reliability')).cpu()
            report['head_weight_changes'][name] = {
                key: float((weights['fusion_module.' + name + '_reliability.' + key]
                            - initializer['fusion_module.' + name + '_reliability.' + key]).norm())
                for key in head.state_dict()}
            with torch.no_grad():
                current = head(test[name]).squeeze(-1).sigmoid()
            variants = {'trained_head': assess(current, test, sensor)}
            for objective in ('l1', 'bce', 'l1_corruption_weighted'):
                variants[objective] = refit(head, fit, test, name, sensor, objective)
                print(name, objective, variants[objective]['validation']['natural_weighted_mae'], flush=True)
            raw_name = 'raw_' + name
            standardized_name = name + '_standardized'
            for split in (fit, test):
                split[standardized_name] = split[name].clone()
            features = (standardized_name, raw_name, name + '_density') if name == 'lidar' else (standardized_name, raw_name)
            for feature in features:
                if feature.endswith('_density'):
                    for split in (fit, test):
                        split['lidar_with_density'] = torch.cat((split['lidar'], split['lidar_density']), -1)
                    feature = 'lidar_with_density'
                mean, std = fit[feature].mean(0), fit[feature].std(0).clamp_min(1e-4)
                fit[feature] = ((fit[feature] - mean) / std).clamp(-20, 20)
                test[feature] = ((test[feature] - mean) / std).clamp(-20, 20)
                variants[feature + '_bce'] = refit(
                    fresh_head(fit[feature].shape[1]), fit, test, feature, sensor, 'bce')
                print(name, feature, variants[feature + '_bce']['validation']['natural_weighted_mae'], flush=True)
            report['head_refits'][name] = variants
            write(args.output / 'diagnosis.json', report)
        assert sha(checkpoint) == completed['checkpoint_sha256']
        report.update(status='complete', checkpoint_unchanged=True,
                      feature_cache_sha256=sha(args.output / 'feature_cache.pt'))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output / 'diagnosis.json', report)


if __name__ == '__main__':
    main()
