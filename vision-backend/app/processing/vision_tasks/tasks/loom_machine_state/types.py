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
    """Motion analysis result for a single loom ROI (MOG2 + rolling buffer)."""

    loom_id: str
    motion_detected: bool  # True if motion_ratio > motion_ratio_stopped (for idle sample buffer)
    motion_ratio: float  # foreground_pixels / total_roi_pixels (0.0–1.0)
    confidence: float  # same as motion_ratio for compatibility
    timestamp: datetime
    frame_index: int
    motion_energy: float = 0.0  # alias for motion_ratio (backward compat)
    optical_flow_magnitude: Optional[float] = None  # unused with MOG2


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
