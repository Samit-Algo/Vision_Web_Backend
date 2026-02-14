"""
Model contracts (protocols)
---------------------------

Defines what a "model" and a "provider" look like. Pipeline calls model(frame);
ModelLoader uses providers to load models by ID.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Optional, Protocol

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
import numpy as np

# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


class Model(Protocol):
    """What a loaded model looks like: callable with (frame, verbose) → results."""

    def __call__(self, frame: np.ndarray, verbose: bool = False) -> Any:
        """Run inference on a BGR frame. Returns model-specific results (e.g. ultralytics Results)."""
        ...


class ModelResult(Protocol):
    """Minimal shape for inference results; we keep original format (e.g. ultralytics) for compatibility."""
    pass


class Provider(Protocol):
    """A provider loads one model by ID (e.g. YOLODetectorProvider loads yolov8n.pt)."""

    def load(self, model_id: str) -> Optional[Model]:
        """Load and return a model, or None if loading fails."""
        ...
