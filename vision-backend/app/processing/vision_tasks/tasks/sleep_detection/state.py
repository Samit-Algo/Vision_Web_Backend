"""
Sleep Detection State
--------------------

We keep a small buffer of recent frames and track:
- pending_analyses: persons that look "possibly sleeping" (waiting for next frame to call VLM)
- deferred_vlm: (analysis, frame_index) — we will call VLM on the *next* frame with 3 frames
- vlm_cache, last_vlm_call_time: so we don't spam the VLM for the same person
- emitted_events: when we last emitted an alert per person (to avoid duplicate alerts)
"""

from typing import List, Dict, Tuple
from datetime import datetime

from app.processing.vision_tasks.tasks.sleep_detection.types import (
    PoseFrame,
    SleepAnalysis,
    SleepVLMConfirmation,
)


class SleepDetectionState:
    """Holds all state for the sleep detection scenario."""

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        # List of recent PoseFrames (oldest first)
        self.pose_buffer: List[PoseFrame] = []
        # Persons that we think might be sleeping (from pose analysis)
        self.pending_analyses: List[SleepAnalysis] = []
        # (analysis, suspicious_frame_index) — call VLM on next frame with [N-1, N, N+1]
        self.deferred_vlm: List[Tuple[SleepAnalysis, int]] = []
        # Cache VLM result per person so we don't call again for a few seconds
        self.vlm_cache: Dict[str, SleepVLMConfirmation] = {}
        # Last time we called VLM for this person (for throttling)
        self.last_vlm_call_time: Dict[str, float] = {}
        # Last time we emitted an alert for this person (to avoid repeating)
        self.emitted_events: Dict[str, datetime] = {}
        # Box for each person we emitted (so UI can draw red "SLEEPING CONFIRMED" even when not in pending_analyses)
        self.emitted_event_boxes: Dict[str, List[float]] = {}
        # person_key -> (first_seen_timestamp, box) when person is "possibly sleeping" AND "still" (for 5s-before-VLM)
        self.person_stable_since: Dict[str, Tuple[datetime, List[float]]] = {}

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
        # Forget emitted events older than 30 seconds (and their boxes)
        self.emitted_events = {
            k: v for k, v in self.emitted_events.items()
            if (current_time - v).total_seconds() < 30.0
        }
        self.emitted_event_boxes = {
            k: v for k, v in self.emitted_event_boxes.items()
            if k in self.emitted_events
        }
        # Forget stable_since older than 30 seconds (person left or moved)
        self.person_stable_since = {
            k: v for k, v in self.person_stable_since.items()
            if (current_time - v[0]).total_seconds() < 30.0
        }

    def reset(self) -> None:
        """Clear everything (e.g. when rule is disabled)."""
        self.pose_buffer.clear()
        self.pending_analyses.clear()
        self.deferred_vlm.clear()
        self.vlm_cache.clear()
        self.last_vlm_call_time.clear()
        self.emitted_events.clear()
        self.emitted_event_boxes.clear()
        self.person_stable_since.clear()
