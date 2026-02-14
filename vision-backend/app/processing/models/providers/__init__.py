"""
Model providers (YOLO detector and pose)
----------------------------------------

Concrete providers that load YOLO models. ModelLookup maps model IDs to these.
"""

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
from app.processing.models.providers.yolo_detector import YOLODetectorProvider
from app.processing.models.providers.yolo_pose import YOLOPoseProvider

__all__ = ["YOLODetectorProvider", "YOLOPoseProvider"]
