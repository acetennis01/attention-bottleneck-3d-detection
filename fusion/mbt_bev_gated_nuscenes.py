"""Six-view gated MBT with explicit view exclusion before height aggregation."""

import torch
from mmdet3d.registry import MODELS

from .mbt_bev import SymmetricMBTBEVFusion as StableFusion
from .mbt_bev_nuscenes import NuScenesSymmetricMBTBEVFusion
from .mbt_bev_gated import GatedCameraMBTBEVFusion


@MODELS.register_module()
class NuScenesGatedCameraMBTBEVFusion(GatedCameraMBTBEVFusion):
    _masked_candidate_softmax = staticmethod(
        StableFusion._masked_candidate_softmax)
    _aligned_image_tokens = NuScenesSymmetricMBTBEVFusion._aligned_image_tokens

    def __init__(self, camera_view_drop_prob=.15, **kwargs):
        if not 0 <= camera_view_drop_prob <= 1:
            raise ValueError('camera_view_drop_prob must be in [0, 1]')
        super().__init__(**kwargs)
        self.camera_view_drop_prob = float(camera_view_drop_prob)

    def _bounded_height_scores(self, features):
        return self.height_score(features)

    def forward(self, image_features, lidar_features, lidar2img=None,
                image_shapes=None, padded_image_shape=None,
                return_attention=False, camera_available=None):
        if image_features.ndim != 5 or image_features.shape[1] != 6:
            raise ValueError('nuScenes requires six-view BVCHW features')
        batch = lidar_features.shape[0]
        if lidar_features.ndim != 4 or image_features.shape[0] != batch:
            raise ValueError('Matching BCHW LiDAR features required')
        if any(x is None for x in (lidar2img, image_shapes, padded_image_shape)):
            raise ValueError('Calibration and image-size metadata required')
        if (lidar2img.shape != (batch, 6, 4, 4) or
                image_shapes.shape != (batch, 6, 2)):
            raise ValueError('Invalid six-view calibration/image-shape dimensions')
        views = torch.ones(batch, 6, dtype=torch.bool,
                           device=image_features.device)
        if camera_available is not None:
            if (camera_available.dtype != torch.bool or
                    camera_available.shape != (batch, 6)):
                raise ValueError('camera_available must be Boolean (B, 6)')
            views = camera_available.to(views.device)
        if self.camera_mode == 'missing':
            views = torch.zeros_like(views)
        if self.training:
            if self.camera_drop_prob:
                views = views & (torch.rand(batch, 1, device=views.device) >=
                                 self.camera_drop_prob)
            if self.camera_view_drop_prob:
                views = views & (torch.rand(batch, 6, device=views.device) >=
                                 self.camera_view_drop_prob)
        # A zero image size invalidates every sampled candidate from that view.
        # This excludes absent views from the normalization denominator, rather
        # than averaging zero features as if they were valid observations.
        shapes = torch.where(views[..., None], image_shapes,
                             torch.zeros_like(image_shapes))
        images = torch.where(views[..., None, None, None], image_features,
                             torch.zeros_like(image_features))
        aligned, valid, weights = self._aligned_image_tokens(
            images, lidar2img, shapes, padded_image_shape,
            return_sampling_weights=True)
        available = views.any(dim=1)
        aligned = torch.where(available[:, None, None], aligned,
                              torch.zeros_like(aligned))
        output = self._fuse_aligned_features(
            lidar_features, aligned, valid, weights,
            return_attention=return_attention,
            sampling_attention_key='image_view_height_weights')
        if self.bypass_missing:
            output['bev_features'] = torch.where(
                available[:, None, None, None], output['bev_features'],
                lidar_features)
        output['camera_available'] = available
        output['camera_view_available'] = views
        return output
