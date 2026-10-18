"""Opt-in image-neck support without changing the historical detector API."""

from torch import Tensor

from mmdet3d.registry import MODELS

from .mbt_detector import MBTBEVFusionDetector


@MODELS.register_module()
class CameraFocusedMBTBEVFusionDetectorV2(MBTBEVFusionDetector):
    """Use a pretrained multi-stage backbone and one selected FPN output."""

    def __init__(self, img_neck: dict, img_feature_level: int = 0,
                 **kwargs) -> None:
        if img_feature_level < 0:
            raise ValueError('img_feature_level must be non-negative')
        super().__init__(**kwargs)
        self.img_neck = MODELS.build(img_neck)
        self.img_feature_level = int(img_feature_level)

    def extract_img_feat(self, images: Tensor) -> Tensor:
        features = self.img_backbone(images)
        if isinstance(features, Tensor):
            features = (features,)
        features = self.img_neck(features)
        if not isinstance(features, (tuple, list)):
            features = (features,)
        if self.img_feature_level >= len(features):
            raise ValueError('img_feature_level exceeds image-neck outputs')
        return features[self.img_feature_level]
