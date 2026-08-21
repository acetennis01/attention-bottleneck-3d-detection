"""Training hooks for the camera-LiDAR fusion experiments."""

from .fusion_diagnostics import FusionDiagnosticsHook
from .staged_fusion_training import StagedFusionTrainingHook

__all__ = ['FusionDiagnosticsHook', 'StagedFusionTrainingHook']
