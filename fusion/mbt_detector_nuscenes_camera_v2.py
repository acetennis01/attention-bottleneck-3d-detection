"""Multi-view ResNet/FPN support with coupled multi-level LiDAR masking."""

import torch.nn.functional as F
from torch import Tensor

from mmdet3d.registry import MODELS

from .mbt_detector_nuscenes import NuScenesMBTBEVFusionDetector


@MODELS.register_module()
class NuScenesCameraFocusedMBTBEVFusionDetectorV2(NuScenesMBTBEVFusionDetector):
    """Preserve the nuScenes detection pyramid and mask every bypass level."""

    def __init__(self, img_neck: dict, img_feature_level: int = 0,
                 **kwargs) -> None:
        if img_feature_level < 0:
            raise ValueError('img_feature_level must be non-negative')
        super().__init__(**kwargs)
        self.img_neck = MODELS.build(img_neck)
        self.img_feature_level = int(img_feature_level)

    def extract_img_feat(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[1] != 6:
            raise ValueError('nuScenes V2 images must be (B, 6, C, H, W)')
        batch_size, views = images.shape[:2]
        flat = images.reshape(batch_size * views, *images.shape[2:])
        features = self.img_backbone(flat)
        if isinstance(features, Tensor):
            features = (features,)
        features = self.img_neck(features)
        if isinstance(features, Tensor):
            features = (features,)
        if self.img_feature_level >= len(features):
            raise ValueError('img_feature_level exceeds image-neck outputs')
        selected = features[self.img_feature_level]
        return selected.reshape(batch_size, views, *selected.shape[1:])

    def extract_feat(self, batch_inputs_dict, batch_data_samples=None,
                     return_fusion_output=False):
        features, output = super().extract_feat(
            batch_inputs_dict, batch_data_samples, return_fusion_output=True)
        features = list(features)
        if self.training and self.fusion_module.lidar_spatial_drop_prob > 0:
            keep = output['lidar_token_keep_mask'].transpose(1, 2).reshape(
                features[0].shape[0], 1, *self.fusion_module.token_grid_size)
            for level, feature in enumerate(features):
                if level == self.fusion_level:
                    continue  # Fusion already masked its bypass and tokens.
                spatial_keep = F.interpolate(
                    keep.float(), size=feature.shape[-2:], mode='nearest').bool()
                features[level] = feature.masked_fill(~spatial_keep, 0.0)
        result = tuple(features)
        return (result, output) if return_fusion_output else result
