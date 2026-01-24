"""
Weapon Detection State Management
----------------------------------

Manages internal state for weapon detection scenario:
- Pose frame buffer
- Pending analyses
- VLM cache and throttling
- Emitted events tracking
"""

from typing import List, Dict
from datetime import datetime

from app.processing.scenarios.weapon_detection.types import (
    PoseFrame,
    ArmPostureAnalysis,
    VLMConfirmation
)


class WeaponDetectionState:
    """Manages state for weapon detection scenario."""
    
    def __init__(self, buffer_size: int):
        """
        Initialize state manager.
        
        Args:
            buffer_size: Maximum number of frames to buffer
        """
        self.buffer_size = buffer_size
        self.pose_buffer: List[PoseFrame] = []
        self.pending_analyses: List[ArmPostureAnalysis] = []
        self.vlm_cache: Dict[str, VLMConfirmation] = {}
        self.last_vlm_call_time: Dict[str, float] = {}
        self.emitted_events: Dict[str, datetime] = {}
    
    def add_pose_frame(self, pose_frame: PoseFrame) -> None:
        """Add pose frame to buffer, removing oldest if buffer is full."""
        self.pose_buffer.append(pose_frame)
        if len(self.pose_buffer) > self.buffer_size:
            self.pose_buffer.pop(0)
    
    def cleanup_old_data(self, current_time: datetime) -> None:
        """
        Remove old pending analyses and emitted events.
        
        Args:
            current_time: Current timestamp for age calculation
        """
        # Remove old pending analyses (older than 5 seconds)
        self.pending_analyses = [
            a for a in self.pending_analyses
            if (current_time - a.timestamp).total_seconds() < 5.0
        ]
        
        # Remove old emitted events (older than 30 seconds)
        self.emitted_events = {
            k: v for k, v in self.emitted_events.items()
            if (current_time - v).total_seconds() < 30.0
        }
    
    def reset(self) -> None:
        """Reset all state."""
        self.pose_buffer.clear()
        self.pending_analyses.clear()
        self.vlm_cache.clear()
        self.last_vlm_call_time.clear()
        self.emitted_events.clear()
