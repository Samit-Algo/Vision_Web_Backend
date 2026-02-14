"""
Model loader (load + cache)
---------------------------

Loads models by ID, resolves aliases, caches instances. Uses ModelLookup to pick
the right provider (YOLO detector vs pose). Default provider is YOLODetectorProvider.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Local
# -----------------------------------------------------------------------------
from app.processing.models.model_lookup import ModelLookup, get_provider
from app.processing.models.data_models import Model, Provider
from app.processing.models.providers.yolo_detector import YOLODetectorProvider
from app.processing.models.providers.yolo_pose import YOLOPoseProvider

# -----------------------------------------------------------------------------
# Aliases (old/invalid names → valid model ID)
# -----------------------------------------------------------------------------

MODEL_ALIASES = {
    "YOLO-Objects-v1": "yolov8m.pt",
    "YOLO-Objects-v2": "yolov8m.pt",
    "YOLO-Objects": "yolov8m.pt",
    "yolo-objects": "yolov8m.pt",
    "YOLO-Detection": "yolov8m.pt",
    "yolo-detection": "yolov8m.pt",
}

_initialized = False


def initialize_default_providers() -> None:
    """Register YOLO detector and pose providers on first use."""
    global _initialized
    if _initialized:
        return
    ModelLookup.register("yolov5", YOLODetectorProvider)
    ModelLookup.register("yolov8", YOLODetectorProvider)
    ModelLookup.register("yolov9", YOLODetectorProvider)
    ModelLookup.register("yolov10", YOLODetectorProvider)
    ModelLookup.register("yolo11", YOLODetectorProvider)
    ModelLookup.register("yolov8-pose", YOLOPoseProvider)
    ModelLookup.register("yolov5-pose", YOLOPoseProvider)
    ModelLookup.register("yolo11-pose", YOLOPoseProvider)
    ModelLookup.register("yolov8n-pose.pt", YOLOPoseProvider)
    ModelLookup.register("yolov8s-pose.pt", YOLOPoseProvider)
    ModelLookup.register("yolov8m-pose.pt", YOLOPoseProvider)
    ModelLookup.register("yolov8l-pose.pt", YOLOPoseProvider)
    ModelLookup.register("yolov8x-pose.pt", YOLOPoseProvider)
    _initialized = True

# -----------------------------------------------------------------------------
# Model loader
# -----------------------------------------------------------------------------


class ModelLoader:
    """
    Load models by ID with caching and alias resolution.

    First call registers default providers. load_model() resolves aliases, checks cache,
    then uses the provider from ModelLookup (or YOLODetectorProvider as default).
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Model] = {}
        initialize_default_providers()

    def load_model(self, model_id: str) -> Optional[Model]:
        """Load one model by ID (alias resolved, then cache, then provider)."""
        original_model_id = model_id
        if model_id in MODEL_ALIASES:
            print(f"[ModelLoader] 🔄 Alias: '{model_id}' -> '{MODEL_ALIASES[model_id]}'")
            model_id = MODEL_ALIASES[model_id]
        if model_id in self._cache:
            return self._cache[model_id]
        provider_class = get_provider(model_id)
        if provider_class is None:
            provider_class = YOLODetectorProvider
        provider: Provider = provider_class()
        model = provider.load(model_id)
        if model is not None:
            self._cache[model_id] = model
            if original_model_id != model_id:
                self._cache[original_model_id] = model
        return model

    def load_models(self, model_ids: List[str]) -> List[Model]:
        """Load multiple models; returns only successful loads."""
        models = []
        for model_id in model_ids:
            m = self.load_model(model_id)
            if m is not None:
                models.append(m)
        return models

    def clear_cache(self) -> None:
        """Clear the model cache (e.g. for memory)."""
        self._cache.clear()

    def get_cached_model(self, model_id: str) -> Optional[Model]:
        """Return cached model if present, without loading."""
        return self._cache.get(model_id)
