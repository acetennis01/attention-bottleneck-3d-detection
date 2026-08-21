"""Geometry-preserving Multimodal Bottleneck Transformer fusion.

This module adapts the symmetric MBT update from Nagrani et al. to camera
and LiDAR feature maps.  Bottleneck tokens mediate communication, while the
full-resolution LiDAR BEV feature map remains the representation consumed by
the 3D detection head.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

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
        point_cloud_range: LiDAR range enabling calibrated image-to-BEV
            sampling. If omitted, the original unaligned tokenization is used.
        height_anchors: LiDAR-frame heights sampled for every BEV token.
        use_local_camera_residual: Whether aligned image content has a direct
            local residual edge into LiDAR tokens. Disable this to require all
            cross-modal semantic exchange to pass through MBT bottlenecks.
        lidar_token_drop_prob: Training-only drop probability for LiDAR tokens
            at camera-visible BEV cells.
        lidar_bev_drop_prob: Training-only channel-drop probability on the raw
            LiDAR BEV bypass.
        camera_aux_num_classes: Number of camera-BEV auxiliary classes. Zero
            disables the bottleneck-conditioned auxiliary head.
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
        point_cloud_range: Optional[Sequence[float]] = None,
        height_anchors: Sequence[float] = (-1.5, -0.75, 0.0, 0.75),
        use_local_camera_residual: bool = True,
        lidar_token_drop_prob: float = 0.0,
        lidar_bev_drop_prob: float = 0.0,
        camera_aux_num_classes: int = 0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        if num_bottleneck_tokens < 1:
            raise ValueError('num_bottleneck_tokens must be positive')
        if num_layers < 1:
            raise ValueError('num_layers must be positive')
        for name, probability in (
                ('lidar_token_drop_prob', lidar_token_drop_prob),
                ('lidar_bev_drop_prob', lidar_bev_drop_prob)):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f'{name} must be in [0, 1)')
        if camera_aux_num_classes < 0:
            raise ValueError('camera_aux_num_classes must be non-negative')

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
        self.use_local_camera_residual = use_local_camera_residual
        self.lidar_token_drop_prob = float(lidar_token_drop_prob)
        self.lidar_bev_drop_prob = float(lidar_bev_drop_prob)
        self.camera_aux_num_classes = int(camera_aux_num_classes)
        self.requires_projection_metadata = point_cloud_range is not None
        if not self.requires_projection_metadata and (
                lidar_token_drop_prob > 0 or camera_aux_num_classes > 0):
            raise ValueError(
                'camera-focused regularization requires geometric alignment')
        if point_cloud_range is not None:
            if len(point_cloud_range) != 6:
                raise ValueError('point_cloud_range must contain six values')
            self.register_buffer(
                'point_cloud_range',
                torch.tensor(point_cloud_range, dtype=torch.float32),
                persistent=False,
            )
            if len(height_anchors) < 1:
                raise ValueError('height_anchors must not be empty')
            self.register_buffer(
                'height_anchors',
                torch.tensor(height_anchors, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.point_cloud_range = None
            self.height_anchors = None

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
        if self.requires_projection_metadata:
            # The two streams now describe the same physical BEV cells, so a
            # shared positional encoding is appropriate.
            self.aligned_bev_position = nn.Parameter(
                torch.empty(1, self.num_spatial_tokens, embed_dim))
            self.height_score = nn.Conv2d(embed_dim, 1, kernel_size=1)
            if use_local_camera_residual:
                self.aligned_image_norm = nn.LayerNorm(embed_dim)
                self.local_image_value = nn.Linear(embed_dim, embed_dim)
                self.local_gate = nn.Sequential(
                    nn.Linear(2 * embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                )
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
        if camera_aux_num_classes > 0:
            # The auxiliary camera prediction is explicitly conditioned on
            # the shared bottlenecks, keeping MBT as the communication path.
            self.camera_readout = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            self.camera_readout_norm = nn.LayerNorm(embed_dim)
            self.camera_aux_head = nn.Linear(
                embed_dim, camera_aux_num_classes)

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
        if self.requires_projection_metadata:
            nn.init.trunc_normal_(self.aligned_bev_position, std=0.02)
            if self.use_local_camera_residual:
                nn.init.zeros_(self.local_gate[-1].weight)
                # Begin with a modest camera contribution
                # (sigmoid(-1) = 0.269).
                nn.init.constant_(self.local_gate[-1].bias, -1.0)
        if self.camera_aux_num_classes > 0:
            nn.init.normal_(self.camera_aux_head.weight, std=0.01)
            # A one-percent foreground prior prevents the dense auxiliary
            # focal loss from being dominated by empty cells at startup.
            nn.init.constant_(self.camera_aux_head.bias, -4.595)
        nn.init.normal_(
            self.bottleneck_tokens, mean=0.0, std=self.bottleneck_init_std)
        nn.init.zeros_(self.bev_residual.weight)
        nn.init.zeros_(self.bev_residual.bias)

    def _tokenize(self, features: Tensor, projection: nn.Module) -> Tensor:
        features = projection(features)
        features = F.adaptive_avg_pool2d(features, self.token_grid_size)
        return features.flatten(2).transpose(1, 2).contiguous()

    def _aligned_image_tokens(
        self,
        image_features: Tensor,
        lidar2img: Tensor,
        image_shapes: Tensor,
        scale_factors: Tensor,
        padded_image_shape: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        """Sample camera features at calibrated 3D BEV query locations."""
        batch_size = image_features.shape[0]
        grid_h, grid_w = self.token_grid_size
        pc_range = self.point_cloud_range.to(image_features)
        heights = self.height_anchors.to(image_features)

        x = torch.linspace(
            pc_range[0], pc_range[3], grid_w + 1,
            device=image_features.device, dtype=image_features.dtype)
        y = torch.linspace(
            pc_range[1], pc_range[4], grid_h + 1,
            device=image_features.device, dtype=image_features.dtype)
        x = 0.5 * (x[:-1] + x[1:])
        y = 0.5 * (y[:-1] + y[1:])
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        num_heights = heights.numel()
        xx = xx.unsqueeze(0).expand(num_heights, -1, -1)
        yy = yy.unsqueeze(0).expand(num_heights, -1, -1)
        zz = heights[:, None, None].expand(-1, grid_h, grid_w)
        ones = torch.ones_like(xx)
        points = torch.stack((xx, yy, zz, ones), dim=-1)
        points = points.reshape(1, -1, 4).expand(batch_size, -1, -1)

        projection = lidar2img.to(
            device=image_features.device, dtype=image_features.dtype)
        projected = torch.bmm(points, projection.transpose(1, 2))
        depth = projected[..., 2]
        safe_depth = depth.clamp_min(1e-5)
        u = projected[..., 0] / safe_depth
        v = projected[..., 1] / safe_depth

        # KITTI's lidar2img matrix is expressed in original-image pixels;
        # Resize records the exact x/y factors applied by the data pipeline.
        factors = scale_factors.to(
            device=image_features.device, dtype=image_features.dtype)
        u = u * factors[:, None, 0]
        v = v * factors[:, None, 1]
        resized_h = image_shapes[:, 0].to(image_features)[:, None]
        resized_w = image_shapes[:, 1].to(image_features)[:, None]
        valid = (
            (depth > 1e-5) & (u >= 0) & (u < resized_w) &
            (v >= 0) & (v < resized_h))

        padded_h, padded_w = padded_image_shape
        # align_corners=False pixel-center normalization.
        grid_x = 2.0 * (u + 0.5) / float(padded_w) - 1.0
        grid_y = 2.0 * (v + 0.5) / float(padded_h) - 1.0
        sample_grid = torch.stack((grid_x, grid_y), dim=-1)
        sample_grid = sample_grid.reshape(
            batch_size, num_heights * grid_h, grid_w, 2)
        sampled = F.grid_sample(
            self.image_projection(image_features),
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch_size, self.embed_dim, num_heights, grid_h, grid_w)
        valid = valid.reshape(
            batch_size, 1, num_heights, grid_h, grid_w)

        scores = self.height_score(
            sampled.permute(0, 2, 1, 3, 4).reshape(
                batch_size * num_heights,
                self.embed_dim, grid_h, grid_w))
        scores = scores.reshape(batch_size, num_heights, grid_h, grid_w)
        scores = scores.masked_fill(~valid[:, 0], -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        aligned = (sampled * weights).sum(dim=2)
        cell_valid = valid.any(dim=2)
        tokens = aligned.flatten(2).transpose(1, 2).contiguous()
        mask = cell_valid.flatten(2).transpose(1, 2).contiguous()
        return tokens, mask

    def forward(
        self,
        image_features: Tensor,
        lidar_features: Tensor,
        lidar2img: Optional[Tensor] = None,
        image_shapes: Optional[Tensor] = None,
        scale_factors: Optional[Tensor] = None,
        padded_image_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Tensor]:
        if image_features.ndim != 4 or lidar_features.ndim != 4:
            raise ValueError('image_features and lidar_features must be BCHW')
        if image_features.shape[0] != lidar_features.shape[0]:
            raise ValueError('camera and LiDAR batch sizes must match')

        batch_size = lidar_features.shape[0]
        lidar_content = self._tokenize(
            lidar_features, self.lidar_projection)
        if self.requires_projection_metadata:
            if any(value is None for value in (
                    lidar2img, image_shapes, scale_factors,
                    padded_image_shape)):
                raise ValueError(
                    'geometry-aligned fusion requires lidar2img, image_shapes, '
                    'scale_factors, and padded_image_shape')
            aligned_image_content, image_valid_mask = (
                self._aligned_image_tokens(
                    image_features, lidar2img, image_shapes, scale_factors,
                    padded_image_shape))
            position = self.aligned_bev_position
            image_tokens = (
                aligned_image_content + position + self.image_modality)
            lidar_token_keep_mask = torch.ones_like(image_valid_mask)
            if self.training and self.lidar_token_drop_prob > 0:
                # Only hide LiDAR cells for which calibrated camera evidence
                # exists; do not ask the camera to reconstruct out-of-FOV BEV.
                drop = (
                    torch.rand_like(lidar_token_keep_mask, dtype=torch.float32)
                    < self.lidar_token_drop_prob) & image_valid_mask
                lidar_token_keep_mask = ~drop
                lidar_content = lidar_content.masked_fill(drop, 0.0)
            lidar_tokens = (
                lidar_content + position + self.lidar_modality)

            if self.use_local_camera_residual:
                # Retained for the earlier aligned experiment. The
                # camera-focused configuration disables this edge so all
                # cross-modal semantic exchange is bottleneck-mediated.
                gate = torch.sigmoid(self.local_gate(torch.cat(
                    (lidar_tokens, image_tokens), dim=-1)))
                lidar_tokens = lidar_tokens + (
                    image_valid_mask.to(lidar_tokens.dtype) * gate *
                    self.local_image_value(
                        self.aligned_image_norm(aligned_image_content)))
        else:
            image_tokens = self._tokenize(
                image_features, self.image_projection)
            image_valid_mask = torch.ones(
                batch_size, self.num_spatial_tokens, 1,
                device=image_features.device, dtype=torch.bool)
            image_tokens = (
                image_tokens + self.image_position + self.image_modality)
            lidar_tokens = (
                lidar_content + self.lidar_position + self.lidar_modality)
            lidar_token_keep_mask = torch.ones_like(image_valid_mask)
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

        if self.camera_aux_num_classes > 0:
            camera_readout, _ = self.camera_readout(
                query=image_tokens,
                key=bottlenecks,
                value=bottlenecks,
                need_weights=False,
            )
            camera_aux_tokens = self.camera_readout_norm(
                image_tokens + camera_readout)
            camera_aux_logits = self.camera_aux_head(camera_aux_tokens)

        token_map = lidar_tokens.transpose(1, 2).reshape(
            batch_size, self.embed_dim, *self.token_grid_size)
        token_map = F.interpolate(
            token_map,
            size=lidar_features.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        lidar_bypass = F.dropout2d(
            lidar_features,
            p=self.lidar_bev_drop_prob,
            training=self.training,
        )
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
            lidar_token_keep_ratio=lidar_token_keep_mask.float().mean(),
        )
        if (self.requires_projection_metadata and
                self.use_local_camera_residual):
            output['local_camera_gate'] = gate
        if self.camera_aux_num_classes > 0:
            output['camera_aux_tokens'] = camera_aux_tokens
            output['camera_aux_logits'] = camera_aux_logits
        return output
