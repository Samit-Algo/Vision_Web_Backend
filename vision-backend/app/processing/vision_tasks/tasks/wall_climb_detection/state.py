"""
Wall Climb Detection State
---------------------------

We keep a small buffer of recent frames and track:
- pose_buffer: recent frames with pose data (for VLM)
- pending_analyses: persons that are above zone (waiting for VLM confirmation)
- deferred_vlm: (analysis, frame_index) — we will call VLM on the *next* frame with 3 frames
- vlm_cache, last_vlm_call_time: so we don't spam the VLM for the same person
- confirmed_violations: track_ids that are VLM-confirmed climbing (for red keypoints)
"""

from typing import List, Dict, Tuple, Set
from datetime import datetime

from app.processing.vision_tasks.tasks.wall_climb_detection.types import (
    PoseFrame,
    WallClimbAnalysis,
    WallClimbVLMConfirmation,
)


class WallClimbDetectionState:
    """Holds all state for the wall climb detection scenario."""

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        # List of recent PoseFrames (oldest first)
        self.pose_buffer: List[PoseFrame] = []
        # Persons that are above zone (from detection analysis)
        self.pending_analyses: List[WallClimbAnalysis] = []
        # (analysis, suspicious_frame_index) — call VLM on next frame with [N-1, N, N+1]
        self.deferred_vlm: List[Tuple[WallClimbAnalysis, int]] = []
        # Cache VLM result per person so we don't call again for a few seconds
        self.vlm_cache: Dict[int, WallClimbVLMConfirmation] = {}  # track_id -> confirmation
        # Last time we called VLM for this person (for throttling)
        self.last_vlm_call_time: Dict[int, float] = {}  # track_id -> timestamp
        # Track IDs that are VLM-confirmed climbing (for red keypoints, no bounding boxes)
        self.confirmed_violations: Set[int] = set()  # track_ids
        # Map track_id -> detection index for confirmed violations (for UI)
        self.confirmed_track_to_detection: Dict[int, int] = {}  # track_id -> detection_index

    def add_pose_frame(self, pose_frame: PoseFrame) -> None:
        """Add one frame to the buffer. Drop the oldest if buffer is full."""
        self.pose_buffer.append(pose_frame)
        if len(self.pose_buffer) > self.buffer_size:
            self.pose_buffer.pop(0)

    def cleanup_old_data(self, current_time: datetime) -> None:
        """Remove old entries so state doesn't grow forever."""
        # Keep only recent pending analyses (e.g. last 5 seconds)
        self.pending_analyses = [
            a for a in self.pending_analyses
            if (current_time - a.timestamp).total_seconds() < 5.0
        ]
        # Same for deferred VLM
        self.deferred_vlm = [
            (a, n) for a, n in self.deferred_vlm
            if (current_time - a.timestamp).total_seconds() < 5.0
        ]
        # Forget confirmed violations older than 30 seconds (person left or moved)
        # We'll clear them when person is no longer above zone

    def reset(self) -> None:
        """Clear everything (e.g. when rule is disabled)."""
        self.pose_buffer.clear()
        self.pending_analyses.clear()
        self.deferred_vlm.clear()
        self.vlm_cache.clear()
        self.last_vlm_call_time.clear()
        self.confirmed_violations.clear()
        self.confirmed_track_to_detection.clear()
