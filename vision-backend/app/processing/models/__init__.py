"""
Models Module
-------------

Provides model loading, caching, and lifecycle management:
- ModelManager: Loads and caches models
- ModelRegistry: Maps model IDs to providers
- Providers: YOLO detector, YOLO pose
"""

from app.processing.models.manager import ModelManager
from app.processing.models.registry import ModelRegistry, register_provider

__all__ = ["ModelManager", "ModelRegistry", "register_provider"]
