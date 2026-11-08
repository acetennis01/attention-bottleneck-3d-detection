"""Unrestricted camera/LiDAR attention control for the Part B KITTI recipe."""

import torch
from torch import nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

from .mbt_bev_gated import GatedCameraMBTBEVFusion


class _FullAttentionBranch(nn.Module):
    """Keep a modality block's parameters; allow keys/values from both streams.

    The query branch's norm and Q/K/V projections apply to both modalities.
    Only that branch's queries are computed: no discarded duplicate outputs.
    """

    def __init__(self, block):
        super().__init__()
        for name in ('norm1', 'attn', 'attn_drop', 'norm2', 'ffn', 'ffn_drop'):
            setattr(self, name, getattr(block, name))

    def forward(self, tokens, context, return_attention=False):
        query = self.norm1(tokens)
        context = self.norm1(context)
        attended, attention = self.attn(
            query, context, context, need_weights=return_attention,
            average_attn_weights=False)
        updated = tokens + self.attn_drop(attended)
        updated = updated + self.ffn_drop(self.ffn(self.norm2(updated)))
        return (updated, attention) if return_attention else updated


@MODELS.register_module()
class FullAttentionBEVFusion(GatedCameraMBTBEVFusion):
    """Two modality-specific branches with unrestricted joint-token attention.

    Reuses KITTI calibration, feature dropout, missing-camera bypass, pre-fusion
    auxiliary supervision and the zero-initialized BEV residual. Every layer's
    two branches read the SAME pre-layer concatenated camera/LiDAR state.
    The final readout queries all final spatial tokens, not shared latents.
    Layer widths/counts and branch/readout parameter shapes match MBT; removal
    of bottleneck embeddings/norm/gates means total parameter counts differ.
    This is full joint attention with modality-specific weights, not the
    alternative single shared-weight transformer or cross-only attention.
    """

    def __init__(self, **kwargs):
        if 'num_bottleneck_tokens' in kwargs or 'use_bottleneck_gate' in kwargs:
            raise ValueError('Full attention has no bottleneck-count/gate option')
        # Consume the same construction RNG as the B8 arm before removing the
        # latent-only tensors. Common learned parameters then start identically
        # under the same seed; this does not add active bottleneck parameters.
        super().__init__(num_bottleneck_tokens=8,
                         use_bottleneck_gate=False, **kwargs)
        self.image_blocks = nn.ModuleList([
            _FullAttentionBranch(block) for block in self.image_blocks])
        self.lidar_blocks = nn.ModuleList([
            _FullAttentionBranch(block) for block in self.lidar_blocks])
        # Do not carry unused bottleneck parameters into counts/checkpoints.
        del self.bottleneck_tokens
        del self.bottleneck_norm
        del self.bottleneck_gates
        del self.num_bottleneck_tokens
        del self.use_bottleneck_gate

    def _fuse_aligned_features(
            self, lidar_features, aligned, image_valid_mask, sampling_weights,
            return_attention=False, sampling_attention_key='image_height_weights'):
        batch = lidar_features.shape[0]
        image = aligned + self.aligned_bev_position + self.image_modality
        aux_tokens = self.camera_aux_norm(image)
        aux_logits = self.camera_aux_head(aux_tokens)
        content, bypass, keep = self._mask_lidar(lidar_features, image_valid_mask)
        lidar = content + self.aligned_bev_position + self.lidar_modality
        image_attention, lidar_attention = [], []
        for image_block, lidar_block in zip(self.image_blocks, self.lidar_blocks):
            context = torch.cat((image, lidar), dim=1)
            next_image = image_block(image, context, return_attention)
            next_lidar = lidar_block(lidar, context, return_attention)
            if return_attention:
                next_image, image_weights = next_image
                next_lidar, lidar_weights = next_lidar
                image_attention.append(image_weights.detach())
                lidar_attention.append(lidar_weights.detach())
            image, lidar = next_image, next_lidar
        image = self.image_norm(image)
        lidar = self.lidar_norm(lidar)
        context = torch.cat((image, lidar), dim=1)
        pre_readout = lidar
        readout, weights = self.lidar_readout(
            query=lidar, key=context, value=context,
            need_weights=return_attention, average_attn_weights=False)
        lidar = self.readout_norm(lidar + readout)
        token_map = lidar.transpose(1, 2).reshape(
            batch, self.embed_dim, *self.token_grid_size)
        token_map = F.interpolate(token_map, size=lidar_features.shape[-2:],
                                  mode='bilinear', align_corners=False)
        sums = sampling_weights.detach().float().reshape(
            batch, -1, *self.token_grid_size).sum(1).flatten(1)
        visible = sums[image_valid_mask[..., 0]]
        output = dict(
            bev_features=bypass + self.bev_residual(token_map),
            image_tokens=image, lidar_tokens=lidar,
            lidar_tokens_pre_readout=pre_readout, fusion_context_tokens=context,
            camera_aux_tokens=aux_tokens, camera_aux_logits=aux_logits,
            image_valid_mask=image_valid_mask,
            image_valid_ratio=image_valid_mask.float().mean(),
            lidar_token_keep_mask=keep, lidar_token_keep_ratio=keep.float().mean(),
            lidar_spatial_drop_prob=lidar_features.new_tensor(0.),
            aligned_image_rms=aligned.detach().float().square().mean().sqrt(),
            camera_weight_sum_min=(visible.min() if visible.numel()
                                   else sums.new_ones(())),
            camera_weight_sum_max=(visible.max() if visible.numel()
                                   else sums.new_ones(())))
        if return_attention:
            output['attention'] = dict(
                image_joint=torch.stack(image_attention, dim=1),
                lidar_joint=torch.stack(lidar_attention, dim=1),
                lidar_readout=weights.detach(),
                **{sampling_attention_key: sampling_weights.detach()})
        return output
