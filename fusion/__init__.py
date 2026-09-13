"""Fusion modules are registered explicitly by configuration."""

__all__ = []
from .mbt_bev_nuscenes import NuScenesSymmetricMBTBEVFusion
from .mbt_detector_nuscenes import NuScenesMBTBEVFusionDetector

__all__ = [
    'NuScenesSymmetricMBTBEVFusion',
    'NuScenesMBTBEVFusionDetector',
]
