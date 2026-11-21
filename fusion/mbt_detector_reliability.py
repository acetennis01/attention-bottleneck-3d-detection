"""Detection and two-sensor severity supervision for the KITTI pilot."""

import torch
from torch.nn import functional as F
from mmdet3d.registry import MODELS

from .mbt_detector_camera_v2 import CameraFocusedMBTBEVFusionDetectorV2


@MODELS.register_module()
class ReliabilityMBTDetector(CameraFocusedMBTBEVFusionDetectorV2):
    def __init__(self, reliability_loss_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        if reliability_loss_weight < 0:
            raise ValueError('reliability_loss_weight must be nonnegative')
        self.reliability_loss_weight = float(reliability_loss_weight)

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        features, output = self.extract_feat(
            batch_inputs_dict, batch_data_samples, return_fusion_output=True)
        losses = self.bbox_head.loss(features, batch_data_samples)
        if self.camera_aux_loss_weight > 0:
            losses['loss_camera_aux'] = (
                self.camera_aux_loss_weight * self._camera_auxiliary_loss(
                    output, batch_data_samples))
        scores = output['modality_quality']
        targets = scores.new_tensor([
            sample.metainfo['modality_quality_target'] for sample in batch_data_samples])
        if targets.shape != scores.shape or not torch.isfinite(targets).all():
            raise ValueError('Expected finite camera/LiDAR targets with shape (B, 2)')
        if not ((targets >= 0) & (targets <= 1)).all():
            raise ValueError('Quality targets must lie in [0, 1]')
        losses['loss_reliability'] = self.reliability_loss_weight * F.l1_loss(
            scores, targets)
        return losses
