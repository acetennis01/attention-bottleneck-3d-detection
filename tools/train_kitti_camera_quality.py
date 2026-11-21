"""Fit a camera-quality head on clean and synthetically degraded KITTI images."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from projects.myfusion.fusion.camera_quality import CameraQualityHead
from projects.myfusion.transforms.quality_corruptions import (
    FIT_CONDITIONS, degrade_camera)


class QualityImages(Dataset):
    def __init__(self, image_root, ids, held_out=False):
        self.image_root = image_root
        self.ids = ids
        self.held_out = held_out

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        path = self.image_root / f'{self.ids[index]}.png'
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape[:2]
        scale = min(1280 / w, 384 / h)
        image = cv2.resize(image, (round(w * scale), round(h * scale)))
        if self.held_out:
            conditions = FIT_CONDITIONS
            strength = 0.65
        else:
            degraded = FIT_CONDITIONS[torch.randint(1, len(FIT_CONDITIONS), ()).item()]
            conditions = ('correct', degraded)
            strength = torch.rand(()).item()
        views = [
            cv2.cvtColor(degrade_camera(image, mode, strength), cv2.COLOR_BGR2RGB)
            for mode in conditions
        ]
        rgb = torch.from_numpy(np.stack(views)).permute(0, 3, 1, 2).float() / 255
        return CameraQualityHead.thumbnail(rgb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cv2.setNumThreads(1)
    train_ids = (args.data_root / 'ImageSets/train.txt').read_text().split()
    validation_ids = set((args.data_root / 'ImageSets/val.txt').read_text().split())
    if (len(train_ids) != 3712 or len(set(train_ids)) != len(train_ids)
            or len(validation_ids) != 3769 or set(train_ids) & validation_ids):
        raise ValueError('Expected disjoint standard KITTI train/validation splits')
    order = np.random.default_rng(0).permutation(train_ids)
    fit_ids, holdout_ids = order[372:], order[:372]
    image_root = args.data_root / 'training/image_2'
    args.work_dir.mkdir(parents=True, exist_ok=False)
    with (args.work_dir / 'split.json').open('w') as stream:
        json.dump(dict(fit=fit_ids.tolist(), holdout=holdout_ids.tolist(),
                       seed=0, conditions=FIT_CONDITIONS), stream)
    fit = DataLoader(QualityImages(image_root, fit_ids), batch_size=24,
                     shuffle=True, num_workers=2, persistent_workers=True)
    holdout = DataLoader(QualityImages(image_root, holdout_ids, held_out=True),
                         batch_size=24, num_workers=1, persistent_workers=True)
    model = CameraQualityHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        fit_loss = 0.0
        for batch in fit:
            target = batch.new_tensor([1.0, 0.0]).expand(len(batch), -1)
            logits = model(batch.flatten(0, 1)).reshape_as(target)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(batch.flatten(0, 1)).view(
                len(batch), len(FIT_CONDITIONS)) for batch in holdout])
            target = torch.zeros_like(logits)
            target[:, 0] = 1
            validation_loss = F.binary_cross_entropy_with_logits(
                logits, target).item()
            weights = logits.sigmoid()
        report = dict(epoch=epoch, fit_bce=fit_loss / len(fit_ids),
                      holdout_bce=validation_loss,
                      mean_weight=dict(zip(FIT_CONDITIONS, weights.mean(0).tolist())),
                      balanced_accuracy=((weights >= 0.5) == target.bool()).float().mean(0).mean().item())
        encoded = json.dumps(report, allow_nan=False)
        if validation_loss < best_loss:
            torch.save(dict(state_dict=model.state_dict(), selection=report),
                       args.work_dir / 'best.tmp')
            (args.work_dir / 'best.tmp').replace(args.work_dir / 'best.pth')
            best_loss = validation_loss
        with (args.work_dir / 'history.jsonl').open('a') as stream:
            stream.write(encoded + '\n')
        print(encoded, flush=True)
    (args.work_dir / 'complete.json').write_text(
        json.dumps(dict(epochs=args.epochs, best_holdout_bce=best_loss)) + '\n')


if __name__ == '__main__':
    main()
