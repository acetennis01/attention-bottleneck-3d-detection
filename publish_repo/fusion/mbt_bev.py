"""Geometry-preserving Multimodal Bottleneck Transformer fusion.

This module adapts the symmetric MBT update from Nagrani et al. to camera
and LiDAR feature maps.  Bottleneck tokens mediate communication, while the
full-resolution LiDAR BEV feature map remains the representation consumed by
the 3D detection head.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


class _ModalityBlock(nn.Module):
    """Pre-norm transformer block for one modality and shared bottlenecks."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        attn_dropout: float,
        proj_dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(proj_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.ffn_drop = nn.Dropout(proj_dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        residual = tokens
        normalized = self.norm1(tokens)
        attended, _ = self.attn(
            normalized, normalized, normalized, need_weights=False)
        tokens = residual + self.attn_drop(attended)
        tokens = tokens + self.ffn_drop(self.ffn(self.norm2(tokens)))
        return tokens


@MODELS.register_module()
class SymmetricMBTBEVFusion(nn.Module):
    """Fuse camera and LiDAR features through symmetric MBT updates.

    At every fusion layer, the image and LiDAR streams have separate
    transformer parameters.  Each stream receives the same bottleneck tokens,
    produces a temporary bottleneck update, and the two temporary updates are
    averaged, matching equations (8) and (9) of the MBT paper.

    The low-resolution fused LiDAR tokens are projected back onto the original
    full-resolution LiDAR BEV map through a zero-initialized residual adapter.
    This preserves the PointPillars geometry and makes the initial model exactly
    equivalent to the LiDAR baseline at the fusion output.

    Args:
        in_img_channels: Channels in the selected image feature map.
        in_lidar_channels: Channels in the LiDAR BEV feature map.
        embed_dim: Common transformer token dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Transformer FFN expansion ratio.
        token_grid_size: Spatial size used to tokenize both feature maps.
        num_bottleneck_tokens: Number of shared fusion tokens.
        num_layers: Number of symmetric MBT fusion layers. At least two are
            recommended because cross-modal information reaches a modality on
            the layer after the bottlenecks first receive it.
        attn_dropout: Attention dropout probability.
        proj_dropout: Residual projection/FFN dropout probability.
        bottleneck_init_std: Standard deviation for bottleneck initialization.
    """

    def __init__(
        self,
        in_img_channels: int = 256,
        in_lidar_channels: int = 384,
        embed_dim: int = 128,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        token_grid_size: Union[int, Sequence[int]] = (16, 16),
        num_bottleneck_tokens: int = 4,
        num_layers: int = 4,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        bottleneck_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        if num_bottleneck_tokens < 1:
            raise ValueError('num_bottleneck_tokens must be positive')
        if num_layers < 1:
            raise ValueError('num_layers must be positive')

        if isinstance(token_grid_size, int):
            token_grid_size = (token_grid_size, token_grid_size)
        if len(token_grid_size) != 2:
            raise ValueError('token_grid_size must contain height and width')
        self.token_grid_size: Tuple[int, int] = (
            int(token_grid_size[0]), int(token_grid_size[1]))
        self.num_spatial_tokens = (
            self.token_grid_size[0] * self.token_grid_size[1])
        self.embed_dim = embed_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens

        self.image_projection = nn.Conv2d(
            in_img_channels, embed_dim, kernel_size=1)
        self.lidar_projection = nn.Conv2d(
            in_lidar_channels, embed_dim, kernel_size=1)

        # Geometry and modality are distinct: never reuse positional parameters
        # between a perspective image grid and a LiDAR BEV grid.
        self.image_position = nn.Parameter(
            torch.empty(1, self.num_spatial_tokens, embed_dim))
        self.lidar_position = nn.Parameter(
            torch.empty(1, self.num_spatial_tokens, embed_dim))
        self.image_modality = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.lidar_modality = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.bottleneck_tokens = nn.Parameter(
            torch.empty(1, num_bottleneck_tokens, embed_dim))

        block_args = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.image_blocks = nn.ModuleList(
            [_ModalityBlock(**block_args) for _ in range(num_layers)])
        self.lidar_blocks = nn.ModuleList(
            [_ModalityBlock(**block_args) for _ in range(num_layers)])
        self.image_norm = nn.LayerNorm(embed_dim)
        self.lidar_norm = nn.LayerNorm(embed_dim)
        self.bottleneck_norm = nn.LayerNorm(embed_dim)
        # The paper classifies both modality streams.  This detector consumes
        # only the LiDAR stream, so an explicit bottleneck-to-LiDAR readout
        # ensures the final image-side bottleneck update reaches the 3D head.
        self.lidar_readout = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.readout_norm = nn.LayerNorm(embed_dim)

        # Retain the standard detector's channel count and spatial resolution.
        # Zero initialization starts training from the LiDAR-only baseline.
        self.bev_residual = nn.Conv2d(
            embed_dim, in_lidar_channels, kernel_size=1, bias=True)

        self.bottleneck_init_std = bottleneck_init_std
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.image_position, std=0.02)
        nn.init.trunc_normal_(self.lidar_position, std=0.02)
        nn.init.trunc_normal_(self.image_modality, std=0.02)
        nn.init.trunc_normal_(self.lidar_modality, std=0.02)
        nn.init.normal_(
            self.bottleneck_tokens, mean=0.0, std=self.bottleneck_init_std)
        nn.init.zeros_(self.bev_residual.weight)
        nn.init.zeros_(self.bev_residual.bias)

    def _tokenize(self, features: Tensor, projection: nn.Module) -> Tensor:
        features = projection(features)
        features = F.adaptive_avg_pool2d(features, self.token_grid_size)
        return features.flatten(2).transpose(1, 2).contiguous()

    def forward(self, image_features: Tensor,
                lidar_features: Tensor) -> Dict[str, Tensor]:
        if image_features.ndim != 4 or lidar_features.ndim != 4:
            raise ValueError('image_features and lidar_features must be BCHW')
        if image_features.shape[0] != lidar_features.shape[0]:
            raise ValueError('camera and LiDAR batch sizes must match')

        batch_size = lidar_features.shape[0]
        image_tokens = self._tokenize(
            image_features, self.image_projection)
        lidar_tokens = self._tokenize(
            lidar_features, self.lidar_projection)
        image_tokens = (
            image_tokens + self.image_position + self.image_modality)
        lidar_tokens = (
            lidar_tokens + self.lidar_position + self.lidar_modality)
        bottlenecks = self.bottleneck_tokens.expand(batch_size, -1, -1)

        for image_block, lidar_block in zip(
                self.image_blocks, self.lidar_blocks):
            image_length = image_tokens.shape[1]
            lidar_length = lidar_tokens.shape[1]

            # Both branches read the same pre-layer bottlenecks.  Their
            # temporary updates are independent and therefore symmetric.
            image_updated = image_block(
                torch.cat((image_tokens, bottlenecks), dim=1))
            lidar_updated = lidar_block(
                torch.cat((lidar_tokens, bottlenecks), dim=1))
            image_tokens = image_updated[:, :image_length]
            lidar_tokens = lidar_updated[:, :lidar_length]
            image_bottlenecks = image_updated[:, image_length:]
            lidar_bottlenecks = lidar_updated[:, lidar_length:]
            bottlenecks = 0.5 * (
                image_bottlenecks + lidar_bottlenecks)

        image_tokens = self.image_norm(image_tokens)
        lidar_tokens = self.lidar_norm(lidar_tokens)
        bottlenecks = self.bottleneck_norm(bottlenecks)
        lidar_tokens_pre_readout = lidar_tokens
        readout, _ = self.lidar_readout(
            query=lidar_tokens,
            key=bottlenecks,
            value=bottlenecks,
            need_weights=False,
        )
        lidar_tokens = self.readout_norm(lidar_tokens + readout)

        token_map = lidar_tokens.transpose(1, 2).reshape(
            batch_size, self.embed_dim, *self.token_grid_size)
        token_map = F.interpolate(
            token_map,
            size=lidar_features.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        fused_bev = lidar_features + self.bev_residual(token_map)

        return dict(
            bev_features=fused_bev,
            image_tokens=image_tokens,
            lidar_tokens=lidar_tokens,
            lidar_tokens_pre_readout=lidar_tokens_pre_readout,
            bottleneck_tokens=bottlenecks,
        )
