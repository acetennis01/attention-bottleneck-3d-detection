"""Read the same shared MBT state at every nuScenes detection scale.

The existing level-0 graph is unchanged. New levels query the final bottleneck
tokens, never image tokens directly. Zero output adapters preserve the entire
old detector at initialization; no historical KITTI implementation is edited.
"""

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

from .mbt_bev_nuscenes_camera_v2 import NuScenesCameraFocusedMBTBEVFusionV2
from .mbt_detector_nuscenes_camera_v2 import (
    NuScenesCameraFocusedMBTBEVFusionDetectorV2)


@MODELS.register_module()
class NuScenesMultiScaleMBTV2(NuScenesCameraFocusedMBTBEVFusionV2):
    """One shared B-token state, with additional LiDAR-query readout scales.

    Upper-level queries do not feed back into the symmetric transformer. This
    deliberately isolates readout coverage rather than changing fusion depth,
    the number of bottlenecks, or the existing level-0 transformer sequence.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.lidar_spatial_drop_prob != 0:
            raise ValueError('Multi-scale control requires hard masking disabled')
        self.pyramid_queries = nn.ModuleDict({
            str(level): deepcopy(self.lidar_projection) for level in (1, 2)})
        # Do not advance the control run's RNG just by constructing adapters.
        with torch.random.fork_rng(devices=[]):
            self.pyramid_residuals = nn.ModuleDict({
                str(level): nn.Conv2d(
                    self.embed_dim, self.bev_residual.out_channels, 1)
                for level in (1, 2)})
        for adapter in self.pyramid_residuals.values():
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    def fuse_pyramid_level(self, feature, bottlenecks, level):
        key = str(level)
        if key not in self.pyramid_queries:
            raise ValueError('Additional pyramid readout is defined for levels 1/2')
        queries = self._tokenize(feature, self.pyramid_queries[key])
        queries = self.lidar_norm(
            queries + self.aligned_bev_position + self.lidar_modality)
        readout, _ = self.lidar_readout(
            query=queries, key=bottlenecks, value=bottlenecks,
            need_weights=False)
        tokens = self.readout_norm(queries + readout)
        token_map = tokens.transpose(1, 2).reshape(
            feature.shape[0], self.embed_dim, *self.token_grid_size)
        token_map = F.interpolate(token_map, size=feature.shape[-2:],
                                  mode='bilinear', align_corners=False)
        return feature + self.pyramid_residuals[key](token_map)


@MODELS.register_module()
class NuScenesMultiScaleMBTDetectorV2(
        NuScenesCameraFocusedMBTBEVFusionDetectorV2):
    """Preserve all three head maps, with MBT-mediated camera paths to each."""

    def extract_feat(self, batch_inputs_dict, batch_data_samples=None,
                     return_fusion_output=False):
        features, output = super().extract_feat(
            batch_inputs_dict, batch_data_samples, return_fusion_output=True)
        if self.fusion_level != 0 or len(features) != 3:
            raise ValueError('Multi-scale control requires the three-level nuScenes FPN')
        if self.fusion_module.lidar_spatial_drop_prob != 0:
            raise ValueError('Multi-scale control requires hard masking disabled')
        result = (features[0],) + tuple(
            self.fusion_module.fuse_pyramid_level(
                features[level], output['bottleneck_tokens'], level)
            for level in (1, 2))
        return (result, output) if return_fusion_output else result
