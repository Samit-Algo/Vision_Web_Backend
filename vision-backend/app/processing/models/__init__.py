"""
Models Module
-------------

Provides model loading, caching, and lifecycle management:
- ModelLoader: Loads and caches models
- ModelLookup: Maps model IDs to providers
- Providers: YOLO detector, YOLO pose
"""

from .model_loader import ModelLoader
from .model_lookup import ModelLookup, register_provider

__all__ = ["ModelLoader", "ModelLookup", "register_provider"]
