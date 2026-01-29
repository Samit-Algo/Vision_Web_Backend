"""
Scenario Models
---------------

Defines the base interfaces/types for scenarios:
- BaseScenario: Abstract base class for all scenarios
- ScenarioFrameContext: Per-frame input from pipeline
- ScenarioEvent: Semantic event output from scenarios

These types are minimal - scenarios define their own internal DTOs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from app.processing.detections.contracts import DetectionPacket
from app.processing.pipeline.contracts import RuleMatch
from app.processing.pipeline.context import PipelineContext


@dataclass
class ScenarioFrameContext:
    """Per-frame context provided to scenarios."""

    frame: np.ndarray
    frame_index: int
    timestamp: datetime
    detections: DetectionPacket
    rule_matches: List[RuleMatch]
    pipeline_context: PipelineContext


@dataclass
class ScenarioEvent:
    """Semantic event emitted by scenarios."""

    event_type: str
    label: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    detection_indices: List[int] = None
    rule_index: Optional[int] = None
    timestamp: Optional[datetime] = None
    frame_index: Optional[int] = None

    def __post_init__(self):
        if self.detection_indices is None:
            self.detection_indices = []
        if self.metadata is None:
            self.metadata = {}


class BaseScenario(ABC):
    """Base class for all scenarios."""

    def __init__(self, config: Dict[str, Any], pipeline_context: PipelineContext):
        self.config = config
        self.pipeline_context = pipeline_context
        self.scenario_id = config.get("type", "unknown")
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """Process a single frame."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset scenario state (called on task restart)."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get current scenario state (for debugging/monitoring)."""
        return self._state.copy()

    def requires_yolo_detections(self) -> bool:
        """Whether this scenario needs YOLO detections."""
        return True

    def get_overlay_data(self) -> Optional[Dict[str, Any]]:
        """Optional overlay visualization data for this scenario."""
        return None

"""
Scenario Data Contracts
-----------------------

Defines the base interfaces for scenarios:
- BaseScenario: Abstract base class for all scenarios
- ScenarioFrameContext: Per-frame input from pipeline
- ScenarioEvent: Semantic event output from scenarios

These contracts are minimal - scenarios define their own internal DTOs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from app.processing.detections.contracts import DetectionPacket
from app.processing.pipeline.contracts import RuleMatch
from app.processing.pipeline.context import PipelineContext


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ScenarioFrameContext:
    """
    Per-frame context provided to scenarios.

    Contains only what pipeline provides each frame.
    Scenarios extract what they need from this (pose keypoints, person boxes, etc.).
    """

    frame: np.ndarray
    frame_index: int
    timestamp: datetime
    detections: DetectionPacket
    rule_matches: List[RuleMatch]
    pipeline_context: PipelineContext


@dataclass
class ScenarioEvent:
    """
    Semantic event emitted by scenarios.

    More structured than RuleMatch - includes:
    - Event type (weapon_detected, production_count, person_identified)
    - Confidence/quality metrics
    - Domain-specific metadata
    - Related detection indices
    """

    event_type: str
    label: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    detection_indices: List[int] = None
    rule_index: Optional[int] = None
    timestamp: Optional[datetime] = None
    frame_index: Optional[int] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.detection_indices is None:
            self.detection_indices = []
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# BASE SCENARIO
# ============================================================================

class BaseScenario(ABC):
    """
    Base class for all scenarios.

    Scenarios are stateful processors that:
    - Receive per-frame context
    - Buffer and reason over time
    - Call additional models when needed
    - Emit events when ready (delayed decisions)
    """

    def __init__(self, config: Dict[str, Any], pipeline_context: PipelineContext):
        """Initialize scenario with configuration."""
        self.config = config
        self.pipeline_context = pipeline_context
        self.scenario_id = config.get("type", "unknown")
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process a single frame.

        Returns:
            List of scenario events (may be empty if decision not ready)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset scenario state (called on task restart)."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get current scenario state (for debugging/monitoring)."""
        return self._state.copy()

    def requires_yolo_detections(self) -> bool:
        """
        Indicate whether this scenario requires YOLO object detection.

        Returns:
            True if scenario needs YOLO detections (boxes, classes, scores),
            False if scenario only needs raw frames (e.g., motion detection scenarios)
        """
        return True

    def get_overlay_data(self) -> Optional[Dict[str, Any]]:
        """
        Get overlay visualization data for this scenario (optional).

        Returns:
            Dict with overlay data, or None if scenario doesn't provide overlays.
        """
        return None
