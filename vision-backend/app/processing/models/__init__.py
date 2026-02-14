"""
Models (load and run inference)
------------------------------

- ModelLoader: load and cache models by ID (with aliases).
- ModelLookup: map model IDs to providers (yolov8, yolov8-pose, etc.).
- Providers: YOLODetectorProvider, YOLOPoseProvider.
"""

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
from .model_loader import ModelLoader
from .model_lookup import ModelLookup, register_provider

__all__ = ["ModelLoader", "ModelLookup", "register_provider"]
