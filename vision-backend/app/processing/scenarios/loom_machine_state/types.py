"""
Loom Machine State Types
------------------------

Data Transfer Objects (DTOs) for loom machine state scenario.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LoomContext:
    """Configuration and metadata for a single loom."""

    loom_id: str
    motion_roi: List[int]
    name: str
    position: Optional[str] = None
    line_number: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MotionAnalysis:
    """Motion analysis result for a single loom ROI."""

    loom_id: str
    motion_detected: bool
    motion_energy: float
    confidence: float
    timestamp: datetime
    frame_index: int
    optical_flow_magnitude: Optional[float] = None


@dataclass
class LoomState:
    """Current state of a loom machine."""

    loom_id: str
    current_state: str
    state_duration_seconds: float
    confidence: float
    last_motion_energy: float
    last_updated: datetime
    state_since: Optional[datetime] = None
    last_optical_flow_magnitude: Optional[float] = None
