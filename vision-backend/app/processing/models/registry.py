"""
Model Registry
--------------

Maps model identifiers to model providers.
Allows the system to route model loading requests to the appropriate provider.
"""

from typing import Dict, Optional, Type
from app.processing.models.contracts import Provider


# Registry mapping model IDs/patterns to provider classes
_model_registry: Dict[str, Type[Provider]] = {}


def register_provider(model_pattern: str, provider_class: Type[Provider]) -> None:
    """
    Register a model provider for a specific model pattern.
    
    Args:
        model_pattern: Pattern to match model IDs (e.g., "yolov8", "yolov8n-pose.pt")
        provider_class: Provider class that can load models matching this pattern
    """
    _model_registry[model_pattern] = provider_class


def get_provider(model_id: str) -> Optional[Type[Provider]]:
    """
    Get the appropriate provider for a model ID.
    
    Matching logic:
    1. Exact match in registry
    2. Prefix match (e.g., "yolov8" matches "yolov8n.pt")
    3. Default to YOLODetectorProvider if no match
    
    Args:
        model_id: Model identifier
    
    Returns:
        Provider class or None if no provider found
    """
    # Try exact match first
    if model_id in _model_registry:
        return _model_registry[model_id]
    
    # Try prefix matches (longest match wins)
    best_match = None
    best_length = 0
    
    for pattern, provider_class in _model_registry.items():
        if model_id.startswith(pattern) and len(pattern) > best_length:
            best_match = provider_class
            best_length = len(pattern)
    
    return best_match


class ModelRegistry:
    """
    Model registry singleton.
    
    Provides access to registered model providers.
    """
    
    @staticmethod
    def register(model_pattern: str, provider_class: Type[Provider]) -> None:
        """Register a provider (delegates to register_provider)."""
        register_provider(model_pattern, provider_class)
    
    @staticmethod
    def get_provider(model_id: str) -> Optional[Type[Provider]]:
        """Get provider for model ID (delegates to get_provider)."""
        return get_provider(model_id)
