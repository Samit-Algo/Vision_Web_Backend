"""
Model lookup (registry)
----------------------

Maps model IDs (or prefixes) to provider classes. ModelLoader uses this to choose
YOLODetectorProvider vs YOLOPoseProvider. Exact match first, then longest prefix match.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Dict, Optional, Type

# -----------------------------------------------------------------------------
# Local
# -----------------------------------------------------------------------------
from app.processing.models.data_models import Provider

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

_model_registry: Dict[str, Type[Provider]] = {}


def register_provider(model_pattern: str, provider_class: Type[Provider]) -> None:
    """Register a provider for a pattern (e.g. 'yolov8' or 'yolov8n-pose.pt')."""
    _model_registry[model_pattern] = provider_class


def get_provider_impl(model_id: str) -> Optional[Type[Provider]]:
    """Return provider class: exact match first, then longest prefix match."""
    if model_id in _model_registry:
        return _model_registry[model_id]
    best_match = None
    best_length = 0
    for pattern, provider_class in _model_registry.items():
        if model_id.startswith(pattern) and len(pattern) > best_length:
            best_match = provider_class
            best_length = len(pattern)
    return best_match


def get_provider(model_id: str) -> Optional[Type[Provider]]:
    """Return the provider class for this model ID (public API)."""
    return get_provider_impl(model_id)


class ModelLookup:
    """Static access to the registry."""

    @staticmethod
    def register(model_pattern: str, provider_class: Type[Provider]) -> None:
        register_provider(model_pattern, provider_class)

    @staticmethod
    def get_provider(model_id: str) -> Optional[Type[Provider]]:
        return get_provider_impl(model_id)
