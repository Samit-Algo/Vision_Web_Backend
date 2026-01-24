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
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np

from app.processing.detections.contracts import DetectionPacket
from app.processing.pipeline.contracts import RuleMatch
from app.processing.pipeline.context import PipelineContext


@dataclass
class ScenarioFrameContext:
    """
    Per-frame context provided to scenarios.
    
    Contains only what pipeline provides each frame.
    Scenarios extract what they need from this (pose keypoints, person boxes, etc.).
    """
    # Frame data
    frame: np.ndarray  # Raw frame (for VLM, face recognition, etc.)
    frame_index: int
    timestamp: datetime
    
    # Detection data
    detections: DetectionPacket  # Merged detections (boxes, classes, scores, keypoints)
    
    # Rule matches (if any)
    rule_matches: List[RuleMatch]  # All rule matches from this frame
    
    # Pipeline context (for task config, agent info)
    pipeline_context: PipelineContext  # Full pipeline context


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
    event_type: str  # "weapon_detected", "production_count", "person_identified"
    label: str  # Human-readable label
    confidence: Optional[float] = None  # Confidence score (0.0-1.0)
    metadata: Optional[Dict[str, Any]] = None  # Scenario-specific data
    detection_indices: List[int] = None  # Related detection indices
    rule_index: Optional[int] = None  # Source rule index (if applicable)
    timestamp: Optional[datetime] = None  # When event occurred
    frame_index: Optional[int] = None  # Frame index when event occurred
    
    def __post_init__(self):
        """Initialize default values."""
        if self.detection_indices is None:
            self.detection_indices = []
        if self.metadata is None:
            self.metadata = {}


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
        """
        Initialize scenario with configuration.
        
        Args:
            config: Scenario-specific configuration from task
            pipeline_context: Pipeline context (for agent/camera info)
        """
        self.config = config
        self.pipeline_context = pipeline_context
        self.scenario_id = config.get("type", "unknown")
        # Scenarios maintain their own state
        self._state: Dict[str, Any] = {}
    
    @abstractmethod
    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process a single frame.
        
        Args:
            frame_context: Per-frame context from pipeline
        
        Returns:
            List of scenario events (may be empty if decision not ready)
        
        Note:
            - May buffer frames internally
            - May call additional models (VLM, face recognition)
            - Emits events only when decision is ready
            - Must return quickly even if decision is pending
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset scenario state (called on task restart).
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current scenario state (for debugging/monitoring).
        
        Returns:
            Scenario state dictionary
        """
        return self._state.copy()
