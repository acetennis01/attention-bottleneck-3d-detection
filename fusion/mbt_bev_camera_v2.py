"""Independent camera supervision and coupled LiDAR masking for KITTI.

This is a new experiment, not a replacement for historical checkpoint code.
The calibrated sampler and symmetric transformer blocks are reused from the
versioned KITTI implementation; all semantic exchange remains MBT-mediated.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

from .mbt_bev_kitti_v1 import SymmetricMBTBEVFusion


class _BoundedHeightScore(nn.Module):
    """Keep height selection well scaled without finite invalid sentinels."""

    def __init__(self, channels: int, limit: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.score = nn.Conv2d(channels, 1, 1)
        self.limit = limit

    def forward(self, features: Tensor) -> Tensor:
        return self.limit * torch.tanh(
            self.score(self.norm(features)) / self.limit)


@MODELS.register_module()
class CameraFocusedMBTBEVFusionV2(SymmetricMBTBEVFusion):
    """Camera-only auxiliary task with one mask for both LiDAR paths.

    ``lidar_spatial_drop_prob`` hides camera-visible BEV regions only during
    training. The same sampled mask is applied before LiDAR tokenization and
    to the full-resolution bypass. Token content is masked again after pooling
    because adaptive-pooling bins can overlap nearest-upsampled boundaries.
    No inverted-dropout rescaling is applied. The full LiDAR scan is used at
    evaluation, and a zero-initialized output adapter preserves baseline output
    at initialization when masking is disabled.
    """

    def __init__(
        self,
        lidar_spatial_drop_prob: float = 0.10,
        camera_aux_num_classes: int = 3,
        height_score_limit: float = 10.0,
        **kwargs,
    ) -> None:
        if not 0 <= lidar_spatial_drop_prob < 1:
            raise ValueError('lidar_spatial_drop_prob must be in [0, 1)')
        if camera_aux_num_classes < 1:
            raise ValueError('V2 requires a camera-only auxiliary head')
        if height_score_limit <= 0:
            raise ValueError('height_score_limit must be positive')
        if kwargs.get('point_cloud_range') is None:
            raise ValueError('V2 requires calibrated BEV alignment')
        # Reject accidental direct camera shortcuts or independent LiDAR masks.
        for name, required in (
                ('use_local_camera_residual', False),
                ('lidar_token_drop_prob', 0.0),
                ('lidar_bev_drop_prob', 0.0),
                ('camera_aggregation_mode', 'valid_only')):
            if name in kwargs and kwargs[name] != required:
                raise ValueError(f'V2 requires {name}={required!r}')
            kwargs[name] = required
        # Do not construct the historical LiDAR-conditioned camera readout.
        super().__init__(camera_aux_num_classes=0, **kwargs)
        self.lidar_spatial_drop_prob = float(lidar_spatial_drop_prob)
        self.camera_aux_num_classes = camera_aux_num_classes
        self.aligned_camera_norm = nn.LayerNorm(self.embed_dim)
        self.height_score = _BoundedHeightScore(
            self.embed_dim, float(height_score_limit))
        self.camera_aux_norm = nn.LayerNorm(self.embed_dim)
        self.camera_aux_head = nn.Linear(
            self.embed_dim, camera_aux_num_classes)
        nn.init.normal_(self.camera_aux_head.weight, std=0.01)
        nn.init.constant_(self.camera_aux_head.bias, -4.595)

    def _mask_lidar(
        self, lidar_features: Tensor, image_valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        keep = torch.ones_like(image_valid_mask, dtype=torch.bool)
        if self.training and self.lidar_spatial_drop_prob > 0:
            drop = (torch.rand_like(keep, dtype=torch.float32) <
                    self.lidar_spatial_drop_prob) & image_valid_mask
            keep = ~drop
        spatial_keep = F.interpolate(
            keep.transpose(1, 2).reshape(
                lidar_features.shape[0], 1, *self.token_grid_size).float(),
            size=lidar_features.shape[-2:], mode='nearest').bool()
        bypass = lidar_features.masked_fill(~spatial_keep, 0.0)
        content = self._tokenize(bypass, self.lidar_projection)
        content = content.masked_fill(~keep, 0.0)
        return content, bypass, keep

    def forward(
        self,
        image_features: Tensor,
        lidar_features: Tensor,
        lidar2img: Optional[Tensor] = None,
        image_shapes: Optional[Tensor] = None,
        scale_factors: Optional[Tensor] = None,
        padded_image_shape: Optional[Tuple[int, int]] = None,
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        if image_features.ndim != 4 or lidar_features.ndim != 4:
            raise ValueError('image_features and lidar_features must be BCHW')
        if image_features.shape[0] != lidar_features.shape[0]:
            raise ValueError('camera and LiDAR batch sizes must match')
        if any(value is None for value in (
                lidar2img, image_shapes, scale_factors, padded_image_shape)):
            raise ValueError('V2 requires calibration and image-size metadata')

        batch_size = lidar_features.shape[0]
        aligned, image_valid_mask, height_weights = self._aligned_image_tokens(
            image_features, lidar2img, image_shapes, scale_factors,
            padded_image_shape, return_sampling_weights=True)
        aligned = self.aligned_camera_norm(aligned)
        return self._fuse_aligned_features(
            lidar_features, aligned, image_valid_mask, height_weights,
            return_attention=return_attention)

    def _fuse_aligned_features(
        self, lidar_features: Tensor, aligned: Tensor,
        image_valid_mask: Tensor, sampling_weights: Tensor,
        return_attention: bool = False,
        sampling_attention_key: str = 'image_height_weights',
    ) -> Dict[str, Tensor]:
        """Shared V2 graph after single- or multi-view geometric sampling.

        ``aligned`` has already been normalized exactly once by the caller.
        Keeping this graph shared prevents dataset variants from drifting in
        auxiliary supervision, coupled masking, or bottleneck exchange.
        """
        batch_size = lidar_features.shape[0]
        image_tokens = (
            aligned + self.aligned_bev_position + self.image_modality)
        # Computed before *any* multimodal attention: no dependence on LiDAR,
        # shared bottlenecks, or the random LiDAR masking decision.
        camera_aux_tokens = self.camera_aux_norm(image_tokens)
        camera_aux_logits = self.camera_aux_head(camera_aux_tokens)
        lidar_content, lidar_bypass, keep = self._mask_lidar(
            lidar_features, image_valid_mask)
        lidar_tokens = (
            lidar_content + self.aligned_bev_position + self.lidar_modality)

        bottlenecks = self.bottleneck_tokens.expand(batch_size, -1, -1)
        image_attention_layers, lidar_attention_layers = [], []
        for image_block, lidar_block in zip(
                self.image_blocks, self.lidar_blocks):
            image_length, lidar_length = (
                image_tokens.shape[1], lidar_tokens.shape[1])
            # Both branches must read the SAME pre-layer shared state.
            image_updated = image_block(
                torch.cat((image_tokens, bottlenecks), dim=1),
                return_attention=return_attention)
            lidar_updated = lidar_block(
                torch.cat((lidar_tokens, bottlenecks), dim=1),
                return_attention=return_attention)
            if return_attention:
                image_updated, image_attention = image_updated
                lidar_updated, lidar_attention = lidar_updated
                image_attention_layers.append(
                    image_attention[:, :, image_length:, :image_length].detach())
                lidar_attention_layers.append(
                    lidar_attention[:, :, lidar_length:, :lidar_length].detach())
            image_tokens = image_updated[:, :image_length]
            lidar_tokens = lidar_updated[:, :lidar_length]
            bottlenecks = 0.5 * (
                image_updated[:, image_length:] +
                lidar_updated[:, lidar_length:])

        image_tokens = self.image_norm(image_tokens)
        lidar_tokens = self.lidar_norm(lidar_tokens)
        bottlenecks = self.bottleneck_norm(bottlenecks)
        pre_readout = lidar_tokens
        readout, readout_attention = self.lidar_readout(
            query=lidar_tokens, key=bottlenecks, value=bottlenecks,
            need_weights=return_attention, average_attn_weights=False)
        lidar_tokens = self.readout_norm(lidar_tokens + readout)
        token_map = lidar_tokens.transpose(1, 2).reshape(
            batch_size, self.embed_dim, *self.token_grid_size)
        token_map = F.interpolate(
            token_map, size=lidar_features.shape[-2:], mode='bilinear',
            align_corners=False)
        # Keep the historical unrestricted residual's capacity. In particular,
        # do not cap the camera's possible correction at 0.1 per BEV feature.
        fused = lidar_bypass + self.bev_residual(token_map)
        sums = sampling_weights.detach().float().reshape(
            batch_size, -1, *self.token_grid_size).sum(1).flatten(1)
        visible_sums = sums[image_valid_mask[..., 0]]
        output = dict(
            bev_features=fused,
            image_tokens=image_tokens,
            lidar_tokens=lidar_tokens,
            lidar_tokens_pre_readout=pre_readout,
            bottleneck_tokens=bottlenecks,
            camera_aux_tokens=camera_aux_tokens,
            camera_aux_logits=camera_aux_logits,
            image_valid_mask=image_valid_mask,
            image_valid_ratio=image_valid_mask.float().mean(),
            lidar_token_keep_mask=keep,
            lidar_token_keep_ratio=keep.float().mean(),
            lidar_spatial_drop_prob=lidar_features.new_tensor(
                self.lidar_spatial_drop_prob),
            aligned_image_rms=aligned.detach().float().square().mean().sqrt(),
            camera_weight_sum_min=(visible_sums.min() if visible_sums.numel()
                                   else sums.new_ones(())),
            camera_weight_sum_max=(visible_sums.max() if visible_sums.numel()
                                   else sums.new_ones(())),
        )
        if return_attention:
            output['attention'] = dict(
                image_bottleneck=torch.stack(image_attention_layers, dim=1),
                lidar_bottleneck=torch.stack(lidar_attention_layers, dim=1),
                lidar_readout=readout_attention.detach(),
                **{sampling_attention_key: sampling_weights.detach()})
        return output
