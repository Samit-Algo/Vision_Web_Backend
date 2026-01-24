"""
Model Providers
---------------

Concrete implementations of model providers for different model types.
"""

from app.processing.models.providers.yolo_detector import YOLODetectorProvider
from app.processing.models.providers.yolo_pose import YOLOPoseProvider

__all__ = ["YOLODetectorProvider", "YOLOPoseProvider"]
