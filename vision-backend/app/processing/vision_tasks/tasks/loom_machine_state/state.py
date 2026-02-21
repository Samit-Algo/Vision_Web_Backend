"""
Loom Machine State Management (Industrial-Grade)
------------------------------------------------

- Rolling buffer of motion_ratio (e.g. 15 seconds at 5 FPS).
- Hysteresis: avg > motion_ratio_running -> RUNNING; avg < motion_ratio_stopped -> STOPPED; else keep previous.
- Debounce on state transitions.
- Idle alert when no motion for >= idle_threshold_minutes (with cooldown).
"""

from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.processing.vision_tasks.tasks.loom_machine_state.types import LoomState, MotionAnalysis


class LoomMachineStateManager:
    """Manages state for loom machine state scenario (MOG2 + rolling buffer + hysteresis)."""

    def __init__(self, config, fps: int = 5):
        self.config = config
        self.fps = fps
        # Processed frames per second = frames_per_minute / 60 (we only add samples when we process a frame)
        processed_per_sec = self.config.frames_per_minute / 60.0
        self.debounce_frames = max(1, int(self.config.debounce_seconds * processed_per_sec))
        self.rolling_buffer_maxlen = max(2, int(self.config.rolling_window_seconds * processed_per_sec))
        # Idle window: need this many no-motion samples to cover idle_threshold_minutes (e.g. 2 min * 4 fpm = 8)
        self.idle_sample_maxlen = max(
            1,
            int(self.config.idle_threshold_minutes * self.config.frames_per_minute),
        )

        self.loom_states: Dict[str, LoomState] = {}
        self.motion_ratio_buffer: Dict[str, deque] = {}
        self.pending_transitions: Dict[str, Dict[str, Any]] = {}
        self.motion_samples: Dict[str, deque] = {}
        self.idle_since: Dict[str, Optional[datetime]] = {}
        self.last_idle_alert_time: Dict[str, Optional[datetime]] = {}

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
                last_updated=datetime.now(),
            )
            self.motion_ratio_buffer[loom_id] = deque(maxlen=self.rolling_buffer_maxlen)
            self.pending_transitions[loom_id] = {"target_state": None, "frames_count": 0}
            self.motion_samples[loom_id] = deque(maxlen=self.idle_sample_maxlen)
            self.idle_since[loom_id] = None
            self.last_idle_alert_time[loom_id] = None

    def add_motion_analysis(self, analysis: MotionAnalysis) -> Optional[str]:
        """
        Add motion_ratio to rolling buffer; decide RUNNING/STOPPED from average with hysteresis.
        Returns previous state if transition occurred.
        """
        loom_id = analysis.loom_id
        if loom_id not in self.loom_states:
            return None

        self.motion_ratio_buffer[loom_id].append((analysis.timestamp, analysis.motion_ratio))
        self.motion_samples[loom_id].append((analysis.timestamp, analysis.motion_detected))

        if analysis.motion_detected:
            self.idle_since[loom_id] = None

        state = self.loom_states[loom_id]
        state.last_motion_energy = getattr(analysis, "motion_ratio", analysis.motion_energy)
        state.last_optical_flow_magnitude = analysis.optical_flow_magnitude
        state.last_updated = analysis.timestamp

        if len(self.motion_ratio_buffer[loom_id]) < 2:
            return None

        buf = list(self.motion_ratio_buffer[loom_id])
        avg_ratio = sum(r for _, r in buf) / len(buf)
        thr_run = self.config.motion_ratio_running
        thr_stop = self.config.motion_ratio_stopped

        if avg_ratio > thr_run:
            target_state = "RUNNING"
        elif avg_ratio < thr_stop:
            target_state = "STOPPED"
        else:
            target_state = state.current_state if state.current_state != "UNKNOWN" else "STOPPED"

        return self._handle_state_transition(loom_id, target_state, analysis.timestamp)

    def _handle_state_transition(
        self, loom_id: str, target_state: str, timestamp: datetime
    ) -> Optional[str]:
        """Debounced state transition. Returns previous state if transition occurred."""
        current_state = self.loom_states[loom_id].current_state
        pending = self.pending_transitions[loom_id]

        if target_state == current_state:
            pending["target_state"] = None
            pending["frames_count"] = 0
            if self.loom_states[loom_id].state_since:
                self.loom_states[loom_id].state_duration_seconds = (
                    timestamp - self.loom_states[loom_id].state_since
                ).total_seconds()
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
                f"[LoomMachineState] Loom '{loom_id}' {previous_state} -> {target_state} @ {timestamp.isoformat()}"
            )
            return previous_state
        return None

    def check_idle_alert(
        self, loom_id: str, loom_name: str, now: datetime
    ) -> Optional[Dict[str, Any]]:
        """Emit idle alert when machine has no motion for >= idle_threshold_minutes (with cooldown)."""
        if not getattr(self.config, "notify_on_idle", True):
            return None
        samples = self.motion_samples.get(loom_id)
        if not samples or len(samples) < self.idle_sample_maxlen:
            return None
        window = list(samples)
        if any(m for _, m in window):
            self.idle_since[loom_id] = None
            return None
        oldest_ts = window[0][0]
        if self.idle_since.get(loom_id) is None:
            self.idle_since[loom_id] = oldest_ts
        idle_duration_minutes = (now - self.idle_since[loom_id]).total_seconds() / 60.0
        if idle_duration_minutes < self.config.idle_threshold_minutes:
            return None
        last_alert = self.last_idle_alert_time.get(loom_id)
        if last_alert is not None:
            if (now - last_alert).total_seconds() < self.config.alert_cooldown_minutes * 60.0:
                return None
        self.last_idle_alert_time[loom_id] = now
        return {
            "loom_id": loom_id,
            "loom_name": loom_name,
            "idle_duration_minutes": idle_duration_minutes,
            "idle_since": self.idle_since[loom_id].isoformat() if self.idle_since.get(loom_id) else None,
        }

    def get_idle_duration_minutes(self, loom_id: str, now: datetime) -> float:
        if self.idle_since.get(loom_id) is None:
            return 0.0
        return (now - self.idle_since[loom_id]).total_seconds() / 60.0

    def get_loom_state(self, loom_id: str) -> LoomState:
        return self.loom_states.get(loom_id)

    def get_all_states(self) -> Dict[str, LoomState]:
        return self.loom_states.copy()

    def cleanup_old_data(self, current_time: datetime) -> None:
        pass

    def reset(self) -> None:
        for loom_id in self.loom_states:
            self.loom_states[loom_id].current_state = "UNKNOWN"
            self.loom_states[loom_id].state_since = None
            self.loom_states[loom_id].state_duration_seconds = 0.0
            self.loom_states[loom_id].confidence = 0.0
            self.motion_ratio_buffer[loom_id].clear()
            self.pending_transitions[loom_id] = {"target_state": None, "frames_count": 0}
            self.motion_samples[loom_id].clear()
            self.idle_since[loom_id] = None
            self.last_idle_alert_time[loom_id] = None
