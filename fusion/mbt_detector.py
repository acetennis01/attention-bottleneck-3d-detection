"""Camera-LiDAR detector using symmetric MBT fusion and a standard 3D head."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from torch import Tensor

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

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        images = batch_inputs_dict.get('imgs')
        voxel_dict = batch_inputs_dict.get('voxels')
        if images is None or voxel_dict is None:
            raise KeyError('MBT fusion requires both imgs and voxels')
        batch_size = int(images.shape[0])
        image_features = self.extract_img_feat(images)
        lidar_features = self.extract_pts_feat(voxel_dict, batch_size)
        fusion_output = self.fusion_module(
            image_features, lidar_features)
        return (fusion_output['bev_features'], )

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: Sequence, **kwargs) -> dict:
        features = self.extract_feat(batch_inputs_dict)
        return self.bbox_head.loss(features, batch_data_samples)

    def predict(self, batch_inputs_dict: dict,
                batch_data_samples: Sequence, **kwargs) -> List:
        features = self.extract_feat(batch_inputs_dict)
        predictions = self.bbox_head.predict(
            features, batch_data_samples, **kwargs)
        return self.add_pred_to_datasample(
            batch_data_samples, predictions)

    def _forward(self, batch_inputs_dict: dict,
                 batch_data_samples: Optional[Sequence] = None,
                 **kwargs):
        features = self.extract_feat(batch_inputs_dict)
        return self.bbox_head.forward(features)
