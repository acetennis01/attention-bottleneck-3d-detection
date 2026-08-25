"""Camera-LiDAR detector using symmetric MBT fusion and a standard 3D head."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS


@MODELS.register_module()
class MBTBEVFusionDetector(Base3DDetector):
    """PointPillars detector enriched by camera features through MBT.

    The detector deliberately keeps the standard PointPillars BEV feature map
    and Anchor3DHead interfaces. Bottlenecks only mediate cross-modal exchange.
    """

    def __init__(
        self,
        img_backbone: dict,
        fusion_module: dict,
        voxel_encoder: dict,
        middle_encoder: dict,
        backbone: dict,
        neck: Optional[dict],
        bbox_head: dict,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        camera_aux_loss_weight: float = 0.0,
        camera_aux_focal_alpha: float = 0.25,
        camera_aux_focal_gamma: float = 2.0,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.img_backbone = MODELS.build(img_backbone)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        self.fusion_module = MODELS.build(fusion_module)

        bbox_head = bbox_head.copy()
        bbox_head.update(train_cfg=train_cfg, test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.camera_aux_loss_weight = float(camera_aux_loss_weight)
        self.camera_aux_focal_alpha = float(camera_aux_focal_alpha)
        self.camera_aux_focal_gamma = float(camera_aux_focal_gamma)
        if self.camera_aux_loss_weight < 0:
            raise ValueError('camera_aux_loss_weight must be non-negative')

    @property
    def with_neck(self) -> bool:
        return self.neck is not None

    def extract_img_feat(self, images: Tensor) -> Tensor:
        features = self.img_backbone(images)
        if isinstance(features, (tuple, list)):
            if len(features) != 1:
                raise ValueError(
                    'img_backbone must expose exactly one feature level')
            features = features[0]
        return features

    def extract_pts_feat(self, voxel_dict: Dict[str, Tensor],
                         batch_size: int) -> Tensor:
        voxel_features = self.voxel_encoder(
            voxel_dict['voxels'],
            voxel_dict['num_points'],
            voxel_dict['coors'],
        )
        features = self.middle_encoder(
            voxel_features, voxel_dict['coors'], batch_size=batch_size)
        features = self.backbone(features)
        if self.with_neck:
            features = self.neck(features)
        if isinstance(features, (tuple, list)):
            if len(features) != 1:
                raise ValueError(
                    'LiDAR neck must return one fused BEV feature level')
            features = features[0]
        return features

    def _projection_metadata(
        self,
        batch_data_samples: Sequence,
        images: Tensor,
    ) -> dict:
        """Collect calibration and resize metadata for aligned BEV sampling."""
        if batch_data_samples is None:
            raise ValueError(
                'geometry-aligned fusion requires batch_data_samples')
        matrices = []
        image_shapes = []
        scale_factors = []
        for sample in batch_data_samples:
            meta = sample.metainfo
            if 'lidar2img' not in meta or 'img_shape' not in meta:
                raise KeyError(
                    'KITTI lidar2img and img_shape metadata are required')
            matrices.append(torch.as_tensor(meta['lidar2img']))
            image_shapes.append(tuple(meta['img_shape'][:2]))
            factor = meta.get('scale_factor', (1.0, 1.0))
            scale_factors.append(tuple(factor[:2]))
        return dict(
            lidar2img=torch.stack(matrices).to(images),
            image_shapes=torch.as_tensor(
                image_shapes, device=images.device),
            scale_factors=torch.as_tensor(
                scale_factors, device=images.device, dtype=images.dtype),
            padded_image_shape=tuple(images.shape[-2:]),
        )

    def extract_feat(
        self,
        batch_inputs_dict: dict,
        batch_data_samples: Optional[Sequence] = None,
        return_fusion_output: bool = False,
    ):
        images = batch_inputs_dict.get('imgs')
        voxel_dict = batch_inputs_dict.get('voxels')
        if images is None or voxel_dict is None:
            raise KeyError('MBT fusion requires both imgs and voxels')
        batch_size = int(images.shape[0])
        image_features = self.extract_img_feat(images)
        lidar_features = self.extract_pts_feat(voxel_dict, batch_size)
        fusion_kwargs = {}
        if getattr(
                self.fusion_module, 'requires_projection_metadata', False):
            fusion_kwargs = self._projection_metadata(
                batch_data_samples, images)
        fusion_output = self.fusion_module(
            image_features, lidar_features, **fusion_kwargs)
        features = (fusion_output['bev_features'], )
        if return_fusion_output:
            return features, fusion_output
        return features

    def _camera_aux_targets(
        self,
        logits: Tensor,
        valid_mask: Tensor,
        batch_data_samples: Sequence,
    ) -> Tuple[Tensor, Tensor]:
        """Create class-specific BEV-center targets for camera supervision."""
        if self.fusion_module.point_cloud_range is None:
            raise RuntimeError('camera auxiliary targets require BEV geometry')
        targets = logits.new_zeros(logits.shape)
        grid_h, grid_w = self.fusion_module.token_grid_size
        pc_range = self.fusion_module.point_cloud_range.to(logits)
        for batch_index, sample in enumerate(batch_data_samples):
            instances = sample.gt_instances_3d
            boxes = instances.bboxes_3d.tensor
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
            grid_x = grid_x[inside]
            grid_y = grid_y[inside]
            labels = labels[inside]
            flat_index = grid_y * grid_w + grid_x
            targets[batch_index, flat_index, labels] = 1.0
        valid = valid_mask.expand_as(logits)
        return targets, valid

    def _camera_auxiliary_loss(
        self,
        fusion_output: dict,
        batch_data_samples: Sequence,
    ) -> Tensor:
        logits = fusion_output['camera_aux_logits']
        targets, valid = self._camera_aux_targets(
            logits, fusion_output['image_valid_mask'], batch_data_samples)
        probabilities = logits.sigmoid()
        alpha = self.camera_aux_focal_alpha
        gamma = self.camera_aux_focal_gamma
        positive = targets.eq(1) & valid
        negative = targets.eq(0) & valid
        positive_loss = -alpha * (
            (1 - probabilities).pow(gamma) *
            F.logsigmoid(logits)) * positive
        negative_loss = -(1 - alpha) * (
            probabilities.pow(gamma) *
            F.logsigmoid(-logits)) * negative
        normalizer = positive.sum().clamp_min(1).to(logits.dtype)
        return (positive_loss.sum() + negative_loss.sum()) / normalizer

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: Sequence, **kwargs) -> dict:
        features, fusion_output = self.extract_feat(
            batch_inputs_dict,
            batch_data_samples,
            return_fusion_output=True,
        )
        losses = self.bbox_head.loss(features, batch_data_samples)
        if self.camera_aux_loss_weight > 0:
            if 'camera_aux_logits' not in fusion_output:
                raise RuntimeError(
                    'camera_aux_loss_weight requires camera_aux_num_classes')
            losses['loss_camera_aux'] = (
                self.camera_aux_loss_weight * self._camera_auxiliary_loss(
                    fusion_output, batch_data_samples))
        return losses

    def predict(self, batch_inputs_dict: dict,
                batch_data_samples: Sequence, **kwargs) -> List:
        features = self.extract_feat(batch_inputs_dict, batch_data_samples)
        predictions = self.bbox_head.predict(
            features, batch_data_samples, **kwargs)
        return self.add_pred_to_datasample(
            batch_data_samples, predictions)

    def _forward(self, batch_inputs_dict: dict,
                 batch_data_samples: Optional[Sequence] = None,
                 **kwargs):
        features = self.extract_feat(batch_inputs_dict, batch_data_samples)
        return self.bbox_head.forward(features)
