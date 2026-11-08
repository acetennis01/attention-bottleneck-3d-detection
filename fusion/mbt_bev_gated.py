"""Experimental KITTI gated bottlenecks; no old checkpoint aliases change.

Camera absence is an explicit signal, never inferred from black pixels. Exact
fallback is a feature identity and requires the detector to remain frozen to
match the original LiDAR checkpoint. Corruption robustness is not guaranteed.
"""

import torch
from torch import nn
from mmdet3d.registry import MODELS

from .mbt_bev_camera_v2 import CameraFocusedMBTBEVFusionV2


@MODELS.register_module()
class GatedCameraMBTBEVFusion(CameraFocusedMBTBEVFusionV2):
    """Per-layer, per-bottleneck scalar gates initialized to equal averaging.

    Dropout acts on aligned camera features, not raw pixels. This prototype is
    single-camera KITTI only; per-view nuScenes dropout needs sampler support.
    ``camera_mode='missing'`` explicitly marks the complete batch absent.
    ``camera_available`` optionally supplies a per-sample Boolean signal.
    With ``bypass_missing=False``, the absent-camera control can learn a
    LiDAR-conditioned residual, measuring extra fusion capacity/training.
    """

    def __init__(self, camera_drop_prob=0.2, use_bottleneck_gate=True,
                 camera_mode='present', bypass_missing=True, **kwargs):
        if not 0 <= camera_drop_prob <= 1:
            raise ValueError('camera_drop_prob must be in [0, 1]')
        if camera_mode not in ('present', 'missing'):
            raise ValueError('camera_mode must be present or missing')
        if kwargs.get('lidar_spatial_drop_prob', 0) != 0:
            raise ValueError('Gated prototype requires intact LiDAR features')
        kwargs['lidar_spatial_drop_prob'] = 0.0
        super().__init__(**kwargs)
        self.camera_drop_prob = float(camera_drop_prob)
        self.use_bottleneck_gate = bool(use_bottleneck_gate)
        self.camera_mode = camera_mode
        self.bypass_missing = bool(bypass_missing)
        # Construct gates in all arms, keeping their parameter budget identical.
        self.bottleneck_gates = nn.ModuleList([
            nn.Linear(2 * self.embed_dim, 1) for _ in self.image_blocks])
        for gate in self.bottleneck_gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def _combine_bottlenecks(self, image, lidar, layer_index):
        if not self.use_bottleneck_gate:
            return super()._combine_bottlenecks(image, lidar, layer_index)
        gate = self.bottleneck_gates[layer_index](
            torch.cat((lidar, image), dim=-1)).sigmoid()
        return gate * image + (1 - gate) * lidar

    def forward(self, image_features, lidar_features, lidar2img=None,
                image_shapes=None, scale_factors=None, padded_image_shape=None,
                return_attention=False, camera_available=None):
        if image_features.ndim != 4 or lidar_features.ndim != 4:
            raise ValueError('Single-camera KITTI BCHW features required')
        batch = lidar_features.shape[0]
        if image_features.shape[0] != batch:
            raise ValueError('camera and LiDAR batch sizes must match')
        if any(x is None for x in (
                lidar2img, image_shapes, scale_factors, padded_image_shape)):
            raise ValueError('Calibration and image-size metadata required')
        available = torch.ones(batch, dtype=torch.bool,
                               device=lidar_features.device)
        if camera_available is not None:
            if (camera_available.dtype != torch.bool or
                    camera_available.shape != (batch,)):
                raise ValueError('camera_available must be Boolean shape (B,)')
            available = camera_available.to(lidar_features.device)
        if self.camera_mode == 'missing':
            available = torch.zeros_like(available)
        if self.training and self.camera_drop_prob:
            available = available & (
                torch.rand(batch, device=available.device) >=
                self.camera_drop_prob)

        # Remove missing features BEFORE the learned alignment/height sampler.
        image_features = torch.where(available[:, None, None, None],
                                     image_features,
                                     torch.zeros_like(image_features))
        aligned, valid, weights = self._aligned_image_tokens(
            image_features, lidar2img, image_shapes, scale_factors,
            padded_image_shape, return_sampling_weights=True)
        aligned = self.aligned_camera_norm(aligned)
        aligned = torch.where(available[:, None, None], aligned,
                              torch.zeros_like(aligned))
        valid = valid & available[:, None, None]
        output = self._fuse_aligned_features(
            lidar_features, aligned, valid, weights,
            return_attention=return_attention)
        if self.bypass_missing:
            output['bev_features'] = torch.where(
                available[:, None, None, None], output['bev_features'],
                lidar_features)
        output['camera_available'] = available
        return output
