"""Geometry-matched concat/add controls for KITTI fusion timing."""

import torch
from torch import nn
from torch.nn import functional as F
from mmdet3d.registry import MODELS

from .mbt_bev_gated import GatedCameraMBTBEVFusion


@MODELS.register_module()
class SimpleBEVFusion(GatedCameraMBTBEVFusion):
    """Combine aligned 16x16 camera/LiDAR features with a 1x1 convolution.

    Uses the same sampling, embedding width, BEV residual projection and
    frozen detector interface. Timing this freshly initialized control does
    not establish trained detection accuracy.
    """

    def __init__(self, combination='concat', **kwargs):
        if combination not in ('concat', 'add'):
            raise ValueError('combination must be concat or add')
        super().__init__(**kwargs)
        self.combination = combination
        channels = self.embed_dim * (2 if combination == 'concat' else 1)
        self.spatial_fusion = nn.Conv2d(channels, self.embed_dim, 1)
        for name in ('image_blocks', 'lidar_blocks', 'bottleneck_tokens',
                     'bottleneck_norm', 'bottleneck_gates', 'lidar_readout',
                     'readout_norm', 'image_norm', 'lidar_norm'):
            delattr(self, name)

    def _fuse_aligned_features(
            self, lidar_features, aligned, image_valid_mask, sampling_weights,
            return_attention=False, sampling_attention_key='image_height_weights',
            residual_weight=None):
        if return_attention:
            raise ValueError('Concat/add fusion has no attention maps')
        batch = len(lidar_features)
        image = aligned + self.aligned_bev_position + self.image_modality
        aux_tokens = self.camera_aux_norm(image)
        content, bypass, keep = self._mask_lidar(lidar_features, image_valid_mask)
        lidar = content + self.aligned_bev_position + self.lidar_modality
        image = image.transpose(1, 2).reshape(batch, self.embed_dim, *self.token_grid_size)
        lidar = lidar.transpose(1, 2).reshape_as(image)
        joint = torch.cat((lidar, image), dim=1) if self.combination == 'concat' else lidar + image
        fused = self.spatial_fusion(joint)
        fused = F.interpolate(fused, size=lidar_features.shape[-2:], mode='bilinear',
                              align_corners=False)
        delta = self.bev_residual(fused)
        if residual_weight is not None:
            delta = delta * residual_weight.to(delta)[:, None, None, None]
        return dict(bev_features=bypass + delta,
                    camera_aux_logits=self.camera_aux_head(aux_tokens),
                    image_valid_mask=image_valid_mask, lidar_token_keep_mask=keep)
