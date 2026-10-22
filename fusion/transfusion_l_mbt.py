"""MBT residual adapter over the official pretrained TransFusion-L detector.

The OpenMMLab BEVFusion model supplies the complete LiDAR path and head.
No camera-to-head shortcut is added. All cross-modal exchange goes through
the existing symmetric MBT implementation. Historical detectors are untouched.
"""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from projects.BEVFusion.bevfusion import BEVFusion

from .mbt_detector import MBTBEVFusionDetector
from .mbt_detector_nuscenes import NuScenesMBTBEVFusionDetector
from .mbt_detector_nuscenes_camera_v2 import (
    NuScenesCameraFocusedMBTBEVFusionDetectorV2)
# Register the unchanged six-camera symmetric MBT graph.
from .mbt_bev_nuscenes_camera_v2 import NuScenesCameraFocusedMBTBEVFusionV2  # noqa: F401


@MODELS.register_module()
class TransFusionLMBTDetector(BEVFusion):
    """Fuse the 512-channel neck map before the pretrained TransFusion head.

    IMPORTANT: BEVFusion's custom voxelizer produces (x,y,z) coordinates;
    its BEV map is B,C,X,Y, unlike our existing B,C,Y,X sampler. Transpose
    into Y,X *only inside fusion*, then transpose back before the untouched
    detection head. A square map would conceal this bug in shape-only tests.
    """

    lidar_module_names = (
        'pts_voxel_layer', 'pts_voxel_encoder', 'pts_middle_encoder',
        'pts_backbone', 'pts_neck', 'bbox_head')

    extract_img_feat = NuScenesCameraFocusedMBTBEVFusionDetectorV2.extract_img_feat
    _compose_lidar2img = staticmethod(NuScenesMBTBEVFusionDetector._compose_lidar2img)
    _projection_metadata = NuScenesMBTBEVFusionDetector._projection_metadata
    _camera_aux_targets = NuScenesMBTBEVFusionDetector._camera_aux_targets
    _camera_auxiliary_loss = MBTBEVFusionDetector._camera_auxiliary_loss

    def __init__(self, fusion_module: dict, img_feature_level: int = 0,
                 freeze_lidar: bool = True, camera_aux_loss_weight: float = .25,
                 camera_aux_focal_alpha: float = .25,
                 camera_aux_focal_gamma: float = 2., **kwargs):
        if kwargs.get('fusion_layer') is not None or kwargs.get('view_transform') is not None:
            raise ValueError('Use MBT only, not BEVFusion ConvFuser/view_transform')
        if camera_aux_loss_weight < 0:
            raise ValueError('camera_aux_loss_weight must be nonnegative')
        super().__init__(**kwargs)
        self.fusion_module = MODELS.build(fusion_module)
        if self.fusion_module.bev_residual.out_channels != 512:
            raise ValueError('The pretrained neck/head contract requires 512 channels')
        if self.fusion_module.lidar_spatial_drop_prob != 0:
            raise ValueError('Preserve the pretrained baseline: hard masking is disabled')
        self.img_feature_level = int(img_feature_level)
        self.freeze_lidar = bool(freeze_lidar)
        self.camera_aux_loss_weight = float(camera_aux_loss_weight)
        self.camera_aux_focal_alpha = float(camera_aux_focal_alpha)
        self.camera_aux_focal_gamma = float(camera_aux_focal_gamma)
        self._set_lidar_frozen()

    def _set_lidar_frozen(self):
        if self.freeze_lidar:
            for name in self.lidar_module_names:
                module = getattr(self, name)
                module.requires_grad_(False)
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Freeze BatchNorm statistics as well as weights, even after val_loop
        # restores model.train(). Frozen head still backpropagates to its input.
        self._set_lidar_frozen()
        return self

    @staticmethod
    def to_sampler_bev(feature: Tensor) -> Tensor:
        return feature.transpose(-1, -2).contiguous()

    @staticmethod
    def to_detector_bev(feature: Tensor) -> Tensor:
        return feature.transpose(-1, -2).contiguous()

    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        return self._extract_fused(batch_inputs_dict, batch_input_metas)[0]

    def _extract_fused(self, inputs, metas):
        # Never call BEVFusion.extract_feat with RGB: it expects depth-LSS and
        # would fuse by a different mechanism. Reuse only its exact LiDAR path.
        with torch.no_grad() if self.freeze_lidar else nullcontext():
            sparse = self.extract_pts_feat(inputs)
            lidar = self.pts_neck(self.pts_backbone(sparse))
        if len(lidar) != 1 or lidar[0].shape[1] != 512:
            raise ValueError('Expected one 512-channel pretrained neck map')
        images = inputs.get('imgs')
        if images is None:
            raise KeyError('Six RGB cameras are required by the MBT adapter')
        samples = [SimpleNamespace(metainfo=meta) for meta in metas]
        output = self.fusion_module(
            self.extract_img_feat(images), self.to_sampler_bev(lidar[0]),
            **self._projection_metadata(samples, images))
        fused = self.to_detector_bev(output['bev_features'])
        return (fused,), output

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        features, output = self._extract_fused(
            batch_inputs_dict, [sample.metainfo for sample in batch_data_samples])
        losses = self.bbox_head.loss(features, batch_data_samples)
        if self.camera_aux_loss_weight:
            losses['loss_camera_aux'] = self.camera_aux_loss_weight * self._camera_auxiliary_loss(
                output, batch_data_samples)
        return losses
