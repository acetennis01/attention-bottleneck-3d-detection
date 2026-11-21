"""Apply a camera-only residual weight to the gated KITTI detector."""

import torch
from mmdet3d.registry import MODELS

from .camera_quality import CameraQualityHead
from .mbt_detector_camera_v2 import CameraFocusedMBTBEVFusionDetectorV2


@MODELS.register_module()
class KittiQualityWeightedDetector(CameraFocusedMBTBEVFusionDetectorV2):
    def __init__(self, use_camera_quality=True, fixed_camera_quality=None, **kwargs):
        super().__init__(**kwargs)
        if fixed_camera_quality is not None and not 0 <= fixed_camera_quality <= 1:
            raise ValueError('fixed_camera_quality must be in [0, 1]')
        self.camera_quality_head = CameraQualityHead()
        self.use_camera_quality = use_camera_quality
        self.fixed_camera_quality = fixed_camera_quality

    def _projection_metadata(self, batch_data_samples, images):
        metadata = super()._projection_metadata(batch_data_samples, images)
        if self.fixed_camera_quality is not None:
            metadata['residual_weight'] = images.new_full(
                (len(images),), self.fixed_camera_quality)
        elif self.use_camera_quality:
            thumbnails = []
            for image, sample in zip(images, batch_data_samples):
                height, width = sample.metainfo['img_shape'][:2]
                image = image[:, :height, :width].float()
                rgb = (image * self.data_preprocessor.std
                       + self.data_preprocessor.mean) / 255
                thumbnails.append(self.camera_quality_head.thumbnail(
                    rgb.clamp(0, 1).unsqueeze(0)))
            logits = self.camera_quality_head(torch.cat(thumbnails, dim=0))
            metadata['residual_weight'] = logits.sigmoid()
        return metadata
