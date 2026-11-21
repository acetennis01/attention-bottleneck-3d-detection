"""Two-sensor quality estimation and explicit reliability-biased bottlenecks."""

import math

import torch
from torch import nn
from mmdet3d.registry import MODELS

from .mbt_bev_gated import GatedCameraMBTBEVFusion


@MODELS.register_module()
class ReliabilityGatedMBTBEVFusion(GatedCameraMBTBEVFusion):
    """Predict camera/LiDAR severity scores from their pre-fusion tokens.

    Targets describe synthetic input degradation, not calibrated detection
    probabilities. Guidance adds log(q_camera/q_lidar) to each learned gate
    logit. Equal scores recover the ordinary trained gate exactly. Sensor
    dropout is disabled for this pilot so its supervision describes the input
    actually processed. Explicit missing-camera evaluation still uses the
    inherited LiDAR identity path.
    """

    def __init__(self, reliability_strength=1.0, **kwargs):
        if not math.isfinite(reliability_strength) or reliability_strength < 0:
            raise ValueError('reliability_strength must be finite and nonnegative')
        if kwargs.get('camera_drop_prob', 0) != 0:
            raise ValueError('Apply supervised sensor corruption in the data pipeline')
        kwargs['camera_drop_prob'] = 0.0
        super().__init__(**kwargs)
        self.reliability_strength = float(reliability_strength)
        self.camera_reliability = self._quality_head()
        self.lidar_reliability = self._quality_head()

    def _quality_head(self):
        head = nn.Sequential(
            nn.Linear(2 * self.embed_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(head[-1].weight)
        nn.init.constant_(head[-1].bias, math.log(9.0))
        return head

    @staticmethod
    def _statistics(tokens, valid):
        weights = valid.to(tokens)
        count = weights.sum(1).clamp_min(1)
        mean = (tokens * weights).sum(1) / count
        variance = ((tokens - mean[:, None]).square() * weights).sum(1) / count
        return torch.cat((mean, (variance + 1e-6).sqrt()), dim=-1)

    def _fuse_aligned_features(
            self, lidar_features, aligned, image_valid_mask, sampling_weights,
            return_attention=False, sampling_attention_key='image_height_weights',
            residual_weight=None):
        lidar = self._tokenize(lidar_features, self.lidar_projection)
        camera_stats = self._statistics(aligned, image_valid_mask)
        lidar_stats = self._statistics(lidar, torch.ones_like(image_valid_mask))
        quality = torch.cat((self.camera_reliability(camera_stats),
                             self.lidar_reliability(lidar_stats)), dim=-1).sigmoid()
        output = super()._fuse_aligned_features(
            lidar_features, aligned, image_valid_mask, sampling_weights,
            return_attention=return_attention,
            sampling_attention_key=sampling_attention_key,
            residual_weight=residual_weight, modality_quality=quality)
        output['modality_quality'] = quality
        return output

    def _combine_bottlenecks(self, image, lidar, layer_index, modality_quality=None):
        logits = self.bottleneck_gates[layer_index](torch.cat((lidar, image), -1))
        quality = modality_quality.clamp_min(1e-4)
        bias = quality[:, 0].log() - quality[:, 1].log()
        gate = (logits + self.reliability_strength * bias[:, None, None]).sigmoid()
        return gate * image + (1 - gate) * lidar
