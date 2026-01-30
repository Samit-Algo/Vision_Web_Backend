"""
Weapon Detection Types
----------------------

Data Transfer Objects (DTOs) for weapon detection scenario.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np


@dataclass
class PoseFrame:
    """Single frame with pose detections."""
    frame: np.ndarray
    keypoints: List[List[List[float]]]  # Pose keypoints per person
    person_boxes: List[List[float]]  # Person bounding boxes [x1, y1, x2, y2]
    timestamp: datetime
    frame_index: int


@dataclass
class ArmPostureAnalysis:
    """Analysis of arm posture from pose keypoints."""
    person_index: int
    box: List[float]
    arm_raised: bool  # True if arm is raised (suspicious for weapon)
    arm_angle: float  # Angle of arm from horizontal (degrees)
    confidence: float  # Confidence in posture analysis (0.0-1.0)
    timestamp: datetime
    frame_index: int


@dataclass
class VLMConfirmation:
    """VLM confirmation result."""
    person_index: int
    box: List[float]
    weapon_detected: bool
    weapon_type: Optional[str]  # "gun", "knife", etc.
    confidence: float  # VLM confidence (0.0-1.0)
    description: Optional[str]  # VLM description of what was detected
    vlm_response: Dict[str, Any]  # Raw VLM response
    timestamp: datetime
    frame_index: int
