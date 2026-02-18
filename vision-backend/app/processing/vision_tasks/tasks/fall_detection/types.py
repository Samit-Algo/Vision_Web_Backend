"""
Fall Detection Types
--------------------

Data structures for the fall detection scenario with VLM confirmation.
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
class FallAnalysis:
    """
    Result of our fall detection check (fall suspected based on pose metrics).
    - track_id: tracking ID for this person
    - person_index: which person in the frame (detection index)
    - box: their bounding box
    - metrics: pose metrics (height, angle, etc.)
    - signals: fall detection signals (hip_drop, torso_horizontal, etc.)
    - confidence: how sure we are (0.0–1.0)
    - timestamp, frame_index: when we analyzed
    """
    track_id: int
    person_index: int
    box: List[float]
    metrics: Dict[str, Any]
    signals: Dict[str, bool]
    confidence: float
    timestamp: datetime
    frame_index: int


@dataclass
class FallVLMConfirmation:
    """
    What the VLM returned after we sent it 9 frames (4 before, suspected, 4 after).
    - fall_detected: True only if VLM says the person is falling/has fallen
    - confidence, description: from VLM
    - vlm_response: raw API response (for debugging)
    """
    track_id: int
    person_index: int
    box: List[float]
    fall_detected: bool
    confidence: float
    description: Optional[str]
    vlm_response: Dict[str, Any]
    timestamp: datetime
    frame_index: int
