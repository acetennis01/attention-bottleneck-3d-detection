"""Six-view adapter for the larger camera-focused MBT V2 graph."""

from typing import Optional, Tuple

from torch import Tensor

from mmdet3d.registry import MODELS

from .mbt_bev import SymmetricMBTBEVFusion as StableFusion
from .mbt_bev_camera_v2 import CameraFocusedMBTBEVFusionV2
from .mbt_bev_nuscenes import NuScenesSymmetricMBTBEVFusion


@MODELS.register_module()
class NuScenesCameraFocusedMBTBEVFusionV2(CameraFocusedMBTBEVFusionV2):
    """Reuse calibrated six-view sampling and the unchanged V2 fusion graph.

    Camera intrinsics already include resizing in the nuScenes pipeline; do
    not apply KITTI scale factors again. Valid view/height weights sum to one
    per visible cell, and all-invalid cells have zero sampling weight.
    """

    _masked_candidate_softmax = staticmethod(
        StableFusion._masked_candidate_softmax)
    _aligned_image_tokens = NuScenesSymmetricMBTBEVFusion._aligned_image_tokens

    def _bounded_height_scores(self, features: Tensor) -> Tensor:
        # V2's scorer already contains GroupNorm and bounded logits.
        return self.height_score(features)

    def forward(
        self, image_features: Tensor, lidar_features: Tensor,
        lidar2img: Optional[Tensor] = None,
        image_shapes: Optional[Tensor] = None,
        padded_image_shape: Optional[Tuple[int, int]] = None,
        return_attention: bool = False,
    ):
        if image_features.ndim != 5 or image_features.shape[1] != 6:
            raise ValueError('nuScenes V2 requires six-view BVCHW features')
        if lidar_features.ndim != 4:
            raise ValueError('LiDAR features must be BCHW')
        if image_features.shape[0] != lidar_features.shape[0]:
            raise ValueError('camera and LiDAR batch sizes must match')
        if any(value is None for value in (
                lidar2img, image_shapes, padded_image_shape)):
            raise ValueError('nuScenes V2 requires calibration and image sizes')
        batch_size = image_features.shape[0]
        if tuple(lidar2img.shape) != (batch_size, 6, 4, 4):
            raise ValueError('lidar2img must have shape (B, 6, 4, 4)')
        if tuple(image_shapes.shape) != (batch_size, 6, 2):
            raise ValueError('image_shapes must have shape (B, 6, 2)')
        aligned, valid, weights = self._aligned_image_tokens(
            image_features, lidar2img, image_shapes, padded_image_shape,
            return_sampling_weights=True)
        # The reused multi-view sampler has already applied aligned_camera_norm.
        return self._fuse_aligned_features(
            lidar_features, aligned, valid, weights,
            return_attention=return_attention,
            sampling_attention_key='image_view_height_weights')
