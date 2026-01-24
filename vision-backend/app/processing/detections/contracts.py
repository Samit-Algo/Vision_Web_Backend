"""
Detection Data Contracts
------------------------

Defines the standardized detection data structure used throughout the pipeline.
This contract is frozen - it defines the stable API between detections and rules.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class DetectionPacket:
    """
    Standardized detection data packet passed to rule engine.
    
    Contains all detection information extracted from model results.
    This structure matches the current detections dict format used by rules.
    """
    classes: List[str]  # Detected class names
    scores: List[float]  # Confidence scores
    boxes: List[List[float]]  # Bounding boxes [x1, y1, x2, y2]
    keypoints: List[List[List[float]]]  # Pose keypoints (if available)
    ts: datetime  # Timestamp
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary format (for backward compatibility with current rule engine).
        
        Returns:
            Dictionary with same structure as current detections dict
        """
        return {
            "classes": self.classes,
            "scores": self.scores,
            "boxes": self.boxes,
            "keypoints": self.keypoints,
            "ts": self.ts,
        }
