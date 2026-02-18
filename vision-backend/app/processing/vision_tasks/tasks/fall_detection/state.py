"""
Fall Detection State
---------------------

We keep a buffer of recent frames and track:
- pose_buffer: recent frames with pose data (for VLM)
- deferred_vlm: (analysis, buffer_len_at_suspect) — we call VLM once buffer has 2+ more frames (buffer-relative, works when frames are skipped)
- vlm_cache, last_vlm_call_time: so we don't spam the VLM for the same person
- confirmed_falls: track_ids that are VLM-confirmed falling (for red keypoints)
"""

from typing import List, Dict, Tuple, Set
from datetime import datetime

from app.processing.vision_tasks.tasks.fall_detection.types import (
    PoseFrame,
    FallAnalysis,
    FallVLMConfirmation,
)


class FallDetectionState:
    """Holds all state for the fall detection scenario with VLM."""

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        # List of recent PoseFrames (oldest first)
        self.pose_buffer: List[PoseFrame] = []
        # (analysis, buffer_len_at_suspect) — call VLM once len(pose_buffer) >= buffer_len + 2 and >= 5 (use last 5 frames; Groq max 5 images)
        self.deferred_vlm: List[Tuple[FallAnalysis, int]] = []
        # Cache VLM result per person so we don't call again for a few seconds
        self.vlm_cache: Dict[int, FallVLMConfirmation] = {}  # track_id -> confirmation
        # Last time we called VLM for this person (for throttling)
        self.last_vlm_call_time: Dict[int, float] = {}  # track_id -> timestamp
        # Track IDs that are VLM-confirmed falling (for red keypoints)
        self.confirmed_falls: Set[int] = set()  # track_ids
        # Map track_id -> detection index for confirmed falls (for UI)
        self.confirmed_track_to_detection: Dict[int, int] = {}  # track_id -> detection_index

    def add_pose_frame(self, pose_frame: PoseFrame) -> None:
        """Add one frame to the buffer. Drop the oldest if buffer is full."""
        self.pose_buffer.append(pose_frame)
        if len(self.pose_buffer) > self.buffer_size:
            self.pose_buffer.pop(0)

    def cleanup_old_data(self, current_time: datetime) -> None:
        """Remove old entries so state doesn't grow forever."""
        # Keep only recent deferred VLM calls (e.g. last 10 seconds)
        self.deferred_vlm = [
            (a, buf_len) for a, buf_len in self.deferred_vlm
            if (current_time - a.timestamp).total_seconds() < 10.0
        ]
        # Remove old cache entries (older than 30 seconds)
        to_remove = []
        for track_id, confirmation in self.vlm_cache.items():
            if (current_time - confirmation.timestamp).total_seconds() > 30.0:
                to_remove.append(track_id)
        for track_id in to_remove:
            self.vlm_cache.pop(track_id, None)

    def reset(self) -> None:
        """Clear everything (e.g. when rule is disabled)."""
        self.pose_buffer.clear()
        self.deferred_vlm.clear()
        self.vlm_cache.clear()
        self.last_vlm_call_time.clear()
        self.confirmed_falls.clear()
        self.confirmed_track_to_detection.clear()
