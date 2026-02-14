"""
Loom Machine State Management
------------------------------

Manages internal state for loom machine state scenario:
- Per-loom state machines (RUNNING/STOPPED)
- Motion history buffers
- State transition tracking
"""

from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.processing.vision_tasks.tasks.loom_machine_state.types import LoomState, MotionAnalysis


# ============================================================================
# STATE MANAGER
# ============================================================================

class LoomMachineStateManager:
    """Manages state for loom machine state scenario."""

    def __init__(self, config, fps: int = 5):
        """Initialize state manager."""
        self.config = config
        self.fps = fps

        self.temporal_consistency_frames = max(
            1,
            int(self.config.temporal_consistency_seconds * self.fps)
        )

        self.debounce_frames = max(
            1,
            int(self.config.debounce_seconds * self.fps)
        )

        self.loom_states: Dict[str, LoomState] = {}
        self.motion_history: Dict[str, deque] = {}
        self.pending_transitions: Dict[str, Dict[str, Any]] = {}

        for loom in config.looms:
            loom_id = loom["loom_id"]
            self.loom_states[loom_id] = LoomState(
                loom_id=loom_id,
                current_state="UNKNOWN",
                state_since=None,
                state_duration_seconds=0.0,
                confidence=0.0,
                last_motion_energy=0.0,
                last_optical_flow_magnitude=None,
                last_updated=datetime.now()
            )
            self.motion_history[loom_id] = deque(maxlen=self.temporal_consistency_frames * 2)
            self.pending_transitions[loom_id] = {
                "target_state": None,
                "frames_count": 0
            }

    def add_motion_analysis(self, analysis: MotionAnalysis) -> Optional[str]:
        """Add motion analysis result and update state. Returns previous state if transition occurred."""
        loom_id = analysis.loom_id

        if loom_id not in self.loom_states:
            return None

        self.motion_history[loom_id].append(analysis)

        state = self.loom_states[loom_id]
        state.last_motion_energy = analysis.motion_energy
        state.last_optical_flow_magnitude = analysis.optical_flow_magnitude
        state.last_updated = analysis.timestamp

        if len(self.motion_history[loom_id]) < self.temporal_consistency_frames:
            return None

        recent_analyses = list(self.motion_history[loom_id])[-self.temporal_consistency_frames:]
        motion_detected_count = sum(1 for a in recent_analyses if a.motion_detected)

        motion_ratio = motion_detected_count / len(recent_analyses) if len(recent_analyses) > 0 else 0.0
        motion_threshold_ratio = self.config.temporal_consistency_ratio

        target_state = "RUNNING" if motion_ratio >= motion_threshold_ratio else "STOPPED"

        return self.handle_state_transition(loom_id, target_state, analysis.timestamp)

    def handle_state_transition(
        self,
        loom_id: str,
        target_state: str,
        timestamp: datetime
    ) -> Optional[str]:
        """Handle state transition with debounce logic. Returns previous state if transition occurred."""
        current_state = self.loom_states[loom_id].current_state
        pending = self.pending_transitions[loom_id]

        if target_state == current_state:
            pending["target_state"] = None
            pending["frames_count"] = 0
            if self.loom_states[loom_id].state_since:
                duration = (timestamp - self.loom_states[loom_id].state_since).total_seconds()
                self.loom_states[loom_id].state_duration_seconds = duration
            return None

        if pending["target_state"] == target_state:
            pending["frames_count"] += 1
        else:
            pending["target_state"] = target_state
            pending["frames_count"] = 1

        if pending["frames_count"] >= self.debounce_frames:
            previous_state = current_state

            self.loom_states[loom_id].current_state = target_state
            self.loom_states[loom_id].state_since = timestamp
            self.loom_states[loom_id].state_duration_seconds = 0.0

            pending["target_state"] = None
            pending["frames_count"] = 0

            print(
                f"[LoomMachineStateScenario] 🔄 Loom '{loom_id}' transitioned from '{previous_state}' to '{target_state}' "
                f"at {timestamp}"
            )

            return previous_state

        return None

    def get_loom_state(self, loom_id: str) -> LoomState:
        """Get current state for a loom."""
        return self.loom_states.get(loom_id)

    def get_all_states(self) -> Dict[str, LoomState]:
        """Get all loom states."""
        return self.loom_states.copy()

    def cleanup_old_data(self, current_time: datetime) -> None:
        """Remove old motion history data."""
        pass

    def reset(self) -> None:
        """Reset all state."""
        for loom_id in self.loom_states:
            self.loom_states[loom_id].current_state = "UNKNOWN"
            self.loom_states[loom_id].state_since = None
            self.loom_states[loom_id].state_duration_seconds = 0.0
            self.loom_states[loom_id].confidence = 0.0
            self.motion_history[loom_id].clear()
            self.pending_transitions[loom_id] = {
                "target_state": None,
                "frames_count": 0
            }
