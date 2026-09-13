"""Six-camera nuScenes variant of the geometry-aligned MBT fusion block.

The KITTI implementation in :mod:`mbt_bev` remains unchanged.  This module
extends it with masked aggregation across the six nuScenes camera views and
the configured vertical anchors.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from mmdet3d.registry import MODELS

from .mbt_bev import SymmetricMBTBEVFusion


@MODELS.register_module()
class NuScenesSymmetricMBTBEVFusion(SymmetricMBTBEVFusion):
    """Symmetric MBT fusion with calibrated six-camera BEV sampling."""

    def _aligned_image_tokens(
        self,
        image_features: Tensor,
        lidar2img: Tensor,
        image_shapes: Tensor,
        padded_image_shape: Tuple[int, int],
        return_sampling_weights: bool = False,
    ):
        if image_features.ndim != 5:
            raise ValueError('nuScenes image features must be BVCHW')
        batch_size, num_views = image_features.shape[:2]
        if lidar2img.shape[:2] != (batch_size, num_views):
            raise ValueError('lidar2img must have shape (B, V, 4, 4)')

        grid_h, grid_w = self.token_grid_size
        pc_range = self.point_cloud_range.to(image_features)
        heights = self.height_anchors.to(image_features)
        num_heights = heights.numel()

        x_edges = torch.linspace(
            pc_range[0], pc_range[3], grid_w + 1,
            device=image_features.device, dtype=image_features.dtype)
        y_edges = torch.linspace(
            pc_range[1], pc_range[4], grid_h + 1,
            device=image_features.device, dtype=image_features.dtype)
        x = 0.5 * (x_edges[:-1] + x_edges[1:])
        y = 0.5 * (y_edges[:-1] + y_edges[1:])
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xx = xx.unsqueeze(0).expand(num_heights, -1, -1)
        yy = yy.unsqueeze(0).expand(num_heights, -1, -1)
        zz = heights[:, None, None].expand(-1, grid_h, grid_w)
        points = torch.stack((xx, yy, zz, torch.ones_like(xx)), dim=-1)
        points = points.reshape(1, 1, -1, 4).expand(
            batch_size, num_views, -1, -1)

        projection = lidar2img.to(
            device=image_features.device, dtype=image_features.dtype)
        projected = torch.matmul(points, projection.transpose(-1, -2))
        depth = projected[..., 2]
        u = projected[..., 0] / depth.clamp_min(1e-5)
        v = projected[..., 1] / depth.clamp_min(1e-5)

        shapes = image_shapes.to(image_features)
        resized_h = shapes[..., 0, None]
        resized_w = shapes[..., 1, None]
        valid = (
            (depth > 1e-5) & (u >= 0) & (u < resized_w) &
            (v >= 0) & (v < resized_h))

        padded_h, padded_w = padded_image_shape
        sample_grid = torch.stack((
            2.0 * (u + 0.5) / float(padded_w) - 1.0,
            2.0 * (v + 0.5) / float(padded_h) - 1.0,
        ), dim=-1)
        sample_grid = sample_grid.reshape(
            batch_size * num_views,
            num_heights * grid_h,
            grid_w,
            2,
        )

        flat_images = image_features.reshape(
            batch_size * num_views, *image_features.shape[2:])
        projected_images = self.image_projection(flat_images)
        sampled = F.grid_sample(
            projected_images,
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch_size, num_views, self.embed_dim,
            num_heights, grid_h, grid_w)
        valid = valid.reshape(
            batch_size, num_views, num_heights, grid_h, grid_w)

        score_input = sampled.permute(0, 1, 3, 2, 4, 5).reshape(
            batch_size * num_views * num_heights,
            self.embed_dim, grid_h, grid_w)
        scores = self.height_score(score_input).reshape(
            batch_size, num_views, num_heights, grid_h, grid_w)
        scores = scores.masked_fill(~valid, -1e4)

        candidate_count = num_views * num_heights
        scores = scores.reshape(
            batch_size, candidate_count, grid_h, grid_w)
        candidate_valid = valid.reshape(
            batch_size, candidate_count, grid_h, grid_w)
        weights = torch.softmax(scores, dim=1)
        weights = weights * candidate_valid.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        sampled = sampled.permute(0, 2, 1, 3, 4, 5).reshape(
            batch_size, self.embed_dim, candidate_count, grid_h, grid_w)
        aligned = (sampled * weights[:, None]).sum(dim=2)
        cell_valid = candidate_valid.any(dim=1, keepdim=True)
        tokens = aligned.flatten(2).transpose(1, 2).contiguous()
        mask = cell_valid.flatten(2).transpose(1, 2).contiguous()
        if return_sampling_weights:
            return tokens, mask, weights.reshape(
                batch_size, num_views, num_heights, grid_h, grid_w)
        return tokens, mask

    def forward(
        self,
        image_features: Tensor,
        lidar_features: Tensor,
        lidar2img: Optional[Tensor] = None,
        image_shapes: Optional[Tensor] = None,
        padded_image_shape: Optional[Tuple[int, int]] = None,
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        if image_features.ndim != 5 or lidar_features.ndim != 4:
            raise ValueError(
                'nuScenes image features must be BVCHW and LiDAR features BCHW')
        if image_features.shape[0] != lidar_features.shape[0]:
            raise ValueError('camera and LiDAR batch sizes must match')
        if not self.requires_projection_metadata:
            raise ValueError('nuScenes fusion requires geometric alignment')
        if any(value is None for value in (
                lidar2img, image_shapes, padded_image_shape)):
            raise ValueError(
                'nuScenes fusion requires lidar2img, image_shapes, and '
                'padded_image_shape')

        batch_size = lidar_features.shape[0]
        lidar_content = self._tokenize(lidar_features, self.lidar_projection)
        aligned_output = self._aligned_image_tokens(
            image_features,
            lidar2img,
            image_shapes,
            padded_image_shape,
            return_sampling_weights=return_attention,
        )
        if return_attention:
            aligned_image_content, image_valid_mask, image_view_weights = (
                aligned_output)
        else:
            aligned_image_content, image_valid_mask = aligned_output

        position = self.aligned_bev_position
        image_tokens = aligned_image_content + position + self.image_modality
        lidar_token_keep_mask = torch.ones_like(image_valid_mask)
        if self.training and self.lidar_token_drop_prob > 0:
            drop = (
                torch.rand_like(lidar_token_keep_mask, dtype=torch.float32)
                < self.lidar_token_drop_prob) & image_valid_mask
            lidar_token_keep_mask = ~drop
            lidar_content = lidar_content.masked_fill(drop, 0.0)
        lidar_tokens = lidar_content + position + self.lidar_modality

        if self.use_local_camera_residual:
            gate = torch.sigmoid(self.local_gate(torch.cat(
                (lidar_tokens, image_tokens), dim=-1)))
            lidar_tokens = lidar_tokens + (
                image_valid_mask.to(lidar_tokens.dtype) * gate *
                self.local_image_value(
                    self.aligned_image_norm(aligned_image_content)))

        bottlenecks = self.bottleneck_tokens.expand(batch_size, -1, -1)
        image_bottleneck_attention = []
        lidar_bottleneck_attention = []
        for image_block, lidar_block in zip(
                self.image_blocks, self.lidar_blocks):
            image_length = image_tokens.shape[1]
            lidar_length = lidar_tokens.shape[1]
            image_output = image_block(
                torch.cat((image_tokens, bottlenecks), dim=1),
                return_attention=return_attention)
            lidar_output = lidar_block(
                torch.cat((lidar_tokens, bottlenecks), dim=1),
                return_attention=return_attention)
            if return_attention:
                image_updated, image_attention = image_output
                lidar_updated, lidar_attention = lidar_output
                image_bottleneck_attention.append(
                    image_attention[:, :, image_length:, :image_length].detach())
                lidar_bottleneck_attention.append(
                    lidar_attention[:, :, lidar_length:, :lidar_length].detach())
            else:
                image_updated = image_output
                lidar_updated = lidar_output
            image_tokens = image_updated[:, :image_length]
            lidar_tokens = lidar_updated[:, :lidar_length]
            bottlenecks = 0.5 * (
                image_updated[:, image_length:] +
                lidar_updated[:, lidar_length:])

        image_tokens = self.image_norm(image_tokens)
        lidar_tokens = self.lidar_norm(lidar_tokens)
        bottlenecks = self.bottleneck_norm(bottlenecks)
        lidar_tokens_pre_readout = lidar_tokens
        readout, lidar_readout_attention = self.lidar_readout(
            query=lidar_tokens,
            key=bottlenecks,
            value=bottlenecks,
            need_weights=return_attention,
            average_attn_weights=False)
        lidar_tokens = self.readout_norm(lidar_tokens + readout)

        if self.camera_aux_num_classes > 0:
            camera_readout, camera_readout_attention = self.camera_readout(
                query=image_tokens,
                key=bottlenecks,
                value=bottlenecks,
                need_weights=return_attention,
                average_attn_weights=False)
            camera_aux_tokens = self.camera_readout_norm(
                image_tokens + camera_readout)
            camera_aux_logits = self.camera_aux_head(camera_aux_tokens)

        token_map = lidar_tokens.transpose(1, 2).reshape(
            batch_size, self.embed_dim, *self.token_grid_size)
        token_map = F.interpolate(
            token_map,
            size=lidar_features.shape[-2:],
            mode='bilinear',
            align_corners=False)
        lidar_bypass = F.dropout2d(
            lidar_features,
            p=self.lidar_bev_drop_prob,
            training=self.training)
        fused_bev = lidar_bypass + self.bev_residual(token_map)

        output = dict(
            bev_features=fused_bev,
            image_tokens=image_tokens,
            lidar_tokens=lidar_tokens,
            lidar_tokens_pre_readout=lidar_tokens_pre_readout,
            bottleneck_tokens=bottlenecks,
            image_valid_mask=image_valid_mask,
            image_valid_ratio=image_valid_mask.float().mean(),
            lidar_token_keep_mask=lidar_token_keep_mask,
            lidar_token_keep_ratio=lidar_token_keep_mask.float().mean())
        if self.use_local_camera_residual:
            output['local_camera_gate'] = gate
        if self.camera_aux_num_classes > 0:
            output['camera_aux_tokens'] = camera_aux_tokens
            output['camera_aux_logits'] = camera_aux_logits
        if return_attention:
            attention = dict(
                image_bottleneck=torch.stack(
                    image_bottleneck_attention, dim=1),
                lidar_bottleneck=torch.stack(
                    lidar_bottleneck_attention, dim=1),
                lidar_readout=lidar_readout_attention.detach(),
                image_view_height_weights=image_view_weights.detach())
            if self.camera_aux_num_classes > 0:
                attention['camera_readout'] = camera_readout_attention.detach()
            output['attention'] = attention
        return output
