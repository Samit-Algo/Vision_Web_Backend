"""
Person Near Machine Types
-------------------------

Data Transfer Objects (DTOs) for person near machine monitoring scenario.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LoomContext:
    """Configuration and metadata for a single loom."""

    loom_id: str
    operator_zone: List[int]  # [x1, y1, x2, y2]
    name: str
    position: Optional[str] = None
    line_number: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PersonPresenceAnalysis:
    """Person presence analysis result for a single loom ROI."""

    loom_id: str
    person_detected: bool  # True if at least one person center is inside operator_zone
    person_count: int  # Number of persons inside zone
    timestamp: datetime
    frame_index: int
    confidence: float = 0.0  # Max confidence of detected persons


@dataclass
class LoomPresenceState:
    """Current presence state of a loom machine."""

    loom_id: str
    current_state: str  # "ATTENDED" or "UNATTENDED"
    state_duration_seconds: float
    confidence: float
    last_seen_time: Optional[datetime]  # Last time person was detected in zone
    absence_start_time: Optional[datetime]  # When absence started (after grace_time)
    last_updated: datetime
    state_since: Optional[datetime] = None
