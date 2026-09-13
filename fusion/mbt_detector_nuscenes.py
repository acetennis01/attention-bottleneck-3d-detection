"""nuScenes six-camera detector using the separate MBT fusion variant."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from mmdet3d.registry import MODELS

from .mbt_detector import MBTBEVFusionDetector


@MODELS.register_module()
class NuScenesMBTBEVFusionDetector(MBTBEVFusionDetector):
    """PointPillars detector with six-view nuScenes MBT fusion.

    The highest-resolution FPN feature is fused and the remaining PointPillars
    pyramid levels are passed to the standard detection head unchanged.
    """

    def __init__(self, fusion_level: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fusion_level = int(fusion_level)

    def extract_img_feat(self, images: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError('nuScenes images must have shape (B, V, C, H, W)')
        batch_size, num_views = images.shape[:2]
        flat_images = images.reshape(batch_size * num_views, *images.shape[2:])
        features = self.img_backbone(flat_images)
        if isinstance(features, (tuple, list)):
            if len(features) != 1:
                raise ValueError(
                    'img_backbone must expose exactly one feature level')
            features = features[0]
        return features.reshape(
            batch_size, num_views, *features.shape[1:])

    def extract_pts_feat(self, voxel_dict: Dict[str, Tensor], batch_size: int):
        voxel_features = self.voxel_encoder(
            voxel_dict['voxels'],
            voxel_dict['num_points'],
            voxel_dict['coors'])
        features = self.middle_encoder(
            voxel_features, voxel_dict['coors'], batch_size=batch_size)
        features = self.backbone(features)
        if self.with_neck:
            features = self.neck(features)
        if not isinstance(features, (tuple, list)):
            features = (features, )
        return tuple(features)

    @staticmethod
    def _compose_lidar2img(cam2img: Tensor, lidar2cam: Tensor) -> Tensor:
        if cam2img.shape[-2:] == (3, 3):
            intrinsic = torch.eye(
                4, device=cam2img.device, dtype=cam2img.dtype)
            intrinsic = intrinsic.expand(*cam2img.shape[:-2], 4, 4).clone()
            intrinsic[..., :3, :3] = cam2img
        elif cam2img.shape[-2:] == (4, 4):
            intrinsic = cam2img
        else:
            raise ValueError('cam2img must end in 3x3 or 4x4')
        return torch.matmul(intrinsic, lidar2cam)

    def _projection_metadata(
        self,
        batch_data_samples: Sequence,
        images: Tensor,
    ) -> dict:
        if batch_data_samples is None:
            raise ValueError(
                'geometry-aligned fusion requires batch_data_samples')
        matrices = []
        image_shapes = []
        num_views = images.shape[1]
        default_shape = tuple(images.shape[-2:])
        for sample in batch_data_samples:
            meta = sample.metainfo
            if 'cam2img' not in meta or 'lidar2cam' not in meta:
                raise KeyError(
                    'nuScenes cam2img and lidar2cam metadata are required')
            cam2img = torch.stack([
                torch.as_tensor(
                    np.asarray(matrix),
                    device=images.device,
                    dtype=images.dtype)
                for matrix in meta['cam2img']
            ])
            lidar2cam = torch.stack([
                torch.as_tensor(
                    np.asarray(matrix),
                    device=images.device,
                    dtype=images.dtype)
                for matrix in meta['lidar2cam']
            ])
            matrices.append(self._compose_lidar2img(cam2img, lidar2cam))

            shapes = meta.get('img_shape')
            if shapes is None:
                shapes = [default_shape] * num_views
            elif (len(shapes) == 2 and
                  all(isinstance(value, (int, float)) for value in shapes)):
                shapes = [shapes] * num_views
            image_shapes.append([
                tuple(shape[:2]) for shape in shapes
            ])
        return dict(
            lidar2img=torch.stack(matrices).to(images),
            image_shapes=torch.as_tensor(
                image_shapes, device=images.device, dtype=images.dtype),
            padded_image_shape=default_shape)

    def _camera_aux_targets(
        self,
        logits: Tensor,
        valid_mask: Tensor,
        batch_data_samples: Sequence,
    ) -> Tuple[Tensor, Tensor]:
        """Create nuScenes center targets from tensor or box containers."""
        targets = logits.new_zeros(logits.shape)
        grid_h, grid_w = self.fusion_module.token_grid_size
        pc_range = self.fusion_module.point_cloud_range.to(logits)
        for batch_index, sample in enumerate(batch_data_samples):
            instances = sample.gt_instances_3d
            boxes = instances.bboxes_3d
            boxes = boxes.tensor if hasattr(boxes, 'tensor') else boxes
            labels = instances.labels_3d
            if boxes.numel() == 0:
                continue
            grid_x = torch.floor(
                (boxes[:, 0] - pc_range[0]) /
                (pc_range[3] - pc_range[0]) * grid_w).long()
            grid_y = torch.floor(
                (boxes[:, 1] - pc_range[1]) /
                (pc_range[4] - pc_range[1]) * grid_h).long()
            inside = (
                (grid_x >= 0) & (grid_x < grid_w) &
                (grid_y >= 0) & (grid_y < grid_h) &
                (labels >= 0) & (labels < logits.shape[-1]))
            flat_index = grid_y[inside] * grid_w + grid_x[inside]
            targets[batch_index, flat_index, labels[inside]] = 1.0
        return targets, valid_mask.expand_as(logits)

    def extract_feat(
        self,
        batch_inputs_dict: dict,
        batch_data_samples: Optional[Sequence] = None,
        return_fusion_output: bool = False,
    ):
        images = batch_inputs_dict.get('imgs')
        voxel_dict = batch_inputs_dict.get('voxels')
        if images is None or voxel_dict is None:
            raise KeyError('nuScenes MBT fusion requires imgs and voxels')
        batch_size = int(images.shape[0])
        image_features = self.extract_img_feat(images)
        lidar_features = list(self.extract_pts_feat(voxel_dict, batch_size))
        if not 0 <= self.fusion_level < len(lidar_features):
            raise IndexError('fusion_level is outside the LiDAR pyramid')
        fusion_output = self.fusion_module(
            image_features,
            lidar_features[self.fusion_level],
            **self._projection_metadata(batch_data_samples, images))
        lidar_features[self.fusion_level] = fusion_output['bev_features']
        features = tuple(lidar_features)
        if return_fusion_output:
            return features, fusion_output
        return features
