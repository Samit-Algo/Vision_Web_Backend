"""
Model Manager
-------------

Manages model loading, caching, and lifecycle.
Provides a single interface for loading models regardless of model type.
"""

from typing import Dict, List, Optional, Any

from app.processing.models.registry import ModelRegistry, get_provider
from app.processing.models.contracts import Model, Provider
from app.processing.models.providers.yolo_detector import YOLODetectorProvider
from app.processing.models.providers.yolo_pose import YOLOPoseProvider


# Auto-register default providers on module import
_initialized = False


def _initialize_default_providers() -> None:
    """Register default model providers."""
    global _initialized
    if _initialized:
        return
    
    # Register YOLO detector provider for standard YOLO models
    ModelRegistry.register("yolov5", YOLODetectorProvider)
    ModelRegistry.register("yolov8", YOLODetectorProvider)
    ModelRegistry.register("yolov9", YOLODetectorProvider)
    ModelRegistry.register("yolov10", YOLODetectorProvider)
    ModelRegistry.register("yolo11", YOLODetectorProvider)
    
    # Register YOLO pose provider for pose models
    ModelRegistry.register("yolov8-pose", YOLOPoseProvider)
    ModelRegistry.register("yolov5-pose", YOLOPoseProvider)
    ModelRegistry.register("yolo11-pose", YOLOPoseProvider)
    
    # Also register with .pt suffix for convenience
    ModelRegistry.register("yolov8n-pose.pt", YOLOPoseProvider)
    ModelRegistry.register("yolov8s-pose.pt", YOLOPoseProvider)
    ModelRegistry.register("yolov8m-pose.pt", YOLOPoseProvider)
    ModelRegistry.register("yolov8l-pose.pt", YOLOPoseProvider)
    ModelRegistry.register("yolov8x-pose.pt", YOLOPoseProvider)
    
    _initialized = True


class ModelManager:
    """
    Manages model loading and caching.
    
    Provides a unified interface for loading models, with automatic caching
    to avoid reloading the same model multiple times.
    """
    
    def __init__(self):
        """Initialize model manager with empty cache."""
        self._cache: Dict[str, Model] = {}
        _initialize_default_providers()
    
    def load_model(self, model_id: str) -> Optional[Model]:
        """
        Load a model by ID, using cache if available.
        
        Args:
            model_id: Model identifier (path or standard model name)
        
        Returns:
            Loaded model instance or None if loading failed
        """
        # Check cache first
        if model_id in self._cache:
            return self._cache[model_id]
        
        # Get appropriate provider
        provider_class = get_provider(model_id)
        
        # Default to YOLODetectorProvider if no provider found
        if provider_class is None:
            provider_class = YOLODetectorProvider
        
        # Create provider instance and load model
        provider: Provider = provider_class()
        model = provider.load(model_id)
        
        if model is not None:
            # Cache the loaded model
            self._cache[model_id] = model
        
        return model
    
    def load_models(self, model_ids: List[str]) -> List[Model]:
        """
        Load multiple models.
        
        Args:
            model_ids: List of model identifiers
        
        Returns:
            List of loaded model instances (only successful loads)
        """
        models = []
        for model_id in model_ids:
            model = self.load_model(model_id)
            if model is not None:
                models.append(model)
        return models
    
    def clear_cache(self) -> None:
        """Clear the model cache (useful for memory management)."""
        self._cache.clear()
    
    def get_cached_model(self, model_id: str) -> Optional[Model]:
        """
        Get a model from cache without loading.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Cached model instance or None if not in cache
        """
        return self._cache.get(model_id)
