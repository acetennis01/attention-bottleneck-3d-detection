"""Camera-only quality predictor for the first KITTI residual-weight pilot."""

import torch
from torch import nn
from torch.nn import functional as F


class CameraQualityHead(nn.Module):
    """Predict one logit per RGB image from a small, fixed-size thumbnail."""

    thumbnail_size = (48, 160)

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(), nn.Linear(256, 1),
        )

    @classmethod
    def thumbnail(cls, rgb):
        """Resize unpadded BCHW RGB values in [0, 1]."""
        return F.interpolate(rgb, size=cls.thumbnail_size, mode='bilinear',
                             align_corners=False, antialias=True)

    def forward(self, thumbnails):
        return self.layers(thumbnails).squeeze(-1)
