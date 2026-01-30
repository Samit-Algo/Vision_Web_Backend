"""
Source Data Contracts
--------------------

Defines the data structures used by sources to pass frame data through the pipeline.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FramePacket:
    """
    Standardized frame data packet passed between pipeline stages.
    
    Contains the frame image and associated metadata needed for processing.
    """
    frame: np.ndarray
    frame_index: int
    timestamp: float  # Monotonic timestamp
    fps: Optional[float] = None  # Source FPS (if available)
    source_id: Optional[str] = None  # Camera ID or file path identifier
    
    def __post_init__(self):
        """Validate frame packet data."""
        if self.frame is None:
            raise ValueError("Frame cannot be None")
        if not isinstance(self.frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
        if len(self.frame.shape) != 3 or self.frame.shape[2] != 3:
            raise ValueError("Frame must be a 3-channel BGR image (H, W, 3)")
