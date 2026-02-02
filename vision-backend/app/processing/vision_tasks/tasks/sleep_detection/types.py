"""
Sleep Detection Types
---------------------

Simple data structures for the sleep detection scenario.
Each type has a short comment so beginners can understand what it holds.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np


@dataclass
class PoseFrame:
    """
    One video frame with pose data for all detected persons.
    - frame: the actual image (for sending to VLM later)
    - keypoints: list of keypoints per person (same order as person_boxes)
    - person_boxes: bounding box [x1, y1, x2, y2] for each person
    - timestamp, frame_index: when this frame was captured
    """
    frame: np.ndarray
    keypoints: List[List[List[float]]]
    person_boxes: List[List[float]]
    timestamp: datetime
    frame_index: int


@dataclass
class SleepAnalysis:
    """
    Result of our simple pose-based "possibly sleeping" check.
    - person_index: which person in the frame
    - box: their bounding box
    - possibly_sleeping: True if pose looks like sleeping (lying or head-down + still)
    - reason: short text like "lying_down" or "head_down_still"
    - confidence: how sure we are (0.0–1.0)
    - timestamp, frame_index: when we analyzed
    """
    person_index: int
    box: List[float]
    possibly_sleeping: bool
    reason: str
    confidence: float
    timestamp: datetime
    frame_index: int


@dataclass
class SleepVLMConfirmation:
    """
    What the VLM returned after we sent it 3 frames.
    - sleeping_detected: True only if VLM says the person is sleeping
    - confidence, description: from VLM
    - vlm_response: raw API response (for debugging)
    """
    person_index: int
    box: List[float]
    sleeping_detected: bool
    confidence: float
    description: Optional[str]
    vlm_response: Dict[str, Any]
    timestamp: datetime
    frame_index: int
