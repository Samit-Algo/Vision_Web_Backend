"""
Frame data contracts
--------------------

Defines the packet passed from sources to the pipeline (Stage 1 → Stage 2).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Optional

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
import numpy as np

# -----------------------------------------------------------------------------
# Data contracts
# -----------------------------------------------------------------------------


@dataclass
class FramePacket:
    """
    One frame plus metadata, passed from source to pipeline.

    Pipeline uses frame for inference and frame_index/timestamp/fps for logging and rules.
    """

    frame: np.ndarray          # BGR image (H, W, 3)
    frame_index: int           # Sequential index from source
    timestamp: float            # Monotonic time (seconds)
    fps: Optional[float] = None       # Source FPS when available
    source_id: Optional[str] = None   # camera_id or video path

    def __post_init__(self) -> None:
        """Validate shape and type of frame."""
        if self.frame is None:
            raise ValueError("Frame cannot be None")
        if not isinstance(self.frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
        if len(self.frame.shape) != 3 or self.frame.shape[2] != 3:
            raise ValueError("Frame must be a 3-channel BGR image (H, W, 3)")
