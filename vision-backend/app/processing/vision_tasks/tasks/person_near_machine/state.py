"""
Person Near Machine State Management (Industrial-Grade)
--------------------------------------------------------

Single zone: ATTENDED when person in polygon for > min_presence_seconds,
UNATTENDED when no person for > grace_time_seconds.
Alert when UNATTENDED for > absence_threshold_minutes (with cooldown).
"""

from datetime import datetime
from typing import Any, Dict, Optional

from app.utils.datetime_utils import utc_now
from app.processing.vision_tasks.tasks.person_near_machine.types import (
    LoomPresenceState,
    PersonPresenceAnalysis,
)

ZONE_ID = "default"  # Single zone for polygon-based rule


class PersonNearMachineStateManager:
    """Manages state for person near machine (single polygon zone)."""

    def __init__(self, config, fps: int = 5):
        self.config = config
        self.fps = fps
        self.grace_time_frames = max(1, int(self.config.grace_time_seconds * fps))
        self.min_presence_frames = max(1, int(self.config.min_presence_seconds * fps))

        self.loom_states: Dict[str, LoomPresenceState] = {
            ZONE_ID: LoomPresenceState(
                loom_id=ZONE_ID,
                current_state="UNKNOWN",
                state_duration_seconds=0.0,
                confidence=0.0,
                last_seen_time=None,
                absence_start_time=None,
                last_updated=utc_now(),
                state_since=None,
            )
        }
        self.presence_buffer: Dict[str, list] = {ZONE_ID: []}
        self.pending_transitions: Dict[str, Dict[str, Any]] = {ZONE_ID: {"target_state": None, "frames_count": 0}}
        self._zone_initialized_at: Dict[str, datetime] = {ZONE_ID: utc_now()}

    def add_presence_analysis(self, analysis: PersonPresenceAnalysis) -> Optional[str]:
        zone_id = analysis.loom_id
        if zone_id not in self.loom_states:
            return None

        state = self.loom_states[zone_id]
        state.last_updated = analysis.timestamp
        if analysis.person_detected:
            state.last_seen_time = analysis.timestamp
            state.confidence = analysis.confidence

        self.presence_buffer[zone_id].append({
            "timestamp": analysis.timestamp,
            "person_detected": analysis.person_detected,
            "person_count": analysis.person_count,
        })
        max_buffer_size = max(self.min_presence_frames * 2, 30)
        if len(self.presence_buffer[zone_id]) > max_buffer_size:
            self.presence_buffer[zone_id] = self.presence_buffer[zone_id][-max_buffer_size:]

        target_state = self._determine_target_state(zone_id, analysis)
        return self._handle_state_transition(zone_id, target_state, analysis.timestamp)

    def _determine_target_state(self, zone_id: str, analysis: PersonPresenceAnalysis) -> str:
        state = self.loom_states[zone_id]
        buffer = self.presence_buffer[zone_id]

        if analysis.person_detected:
            recent_presence = [item["person_detected"] for item in buffer[-self.min_presence_frames:]]
            if len(recent_presence) >= self.min_presence_frames and all(recent_presence):
                return "ATTENDED"
        else:
            if state.last_seen_time is not None:
                if (analysis.timestamp - state.last_seen_time).total_seconds() >= self.config.grace_time_seconds:
                    return "UNATTENDED"
            else:
                # Never seen person: stay UNKNOWN (gray) until grace_time from start, then UNATTENDED (red)
                init_at = self._zone_initialized_at.get(zone_id) or analysis.timestamp
                if (analysis.timestamp - init_at).total_seconds() >= self.config.grace_time_seconds:
                    return "UNATTENDED"
                return "UNKNOWN"
        return state.current_state

    def _handle_state_transition(self, zone_id: str, target_state: str, timestamp: datetime) -> Optional[str]:
        current_state = self.loom_states[zone_id].current_state
        pending = self.pending_transitions[zone_id]

        if target_state == current_state:
            pending["target_state"] = None
            pending["frames_count"] = 0
            if self.loom_states[zone_id].state_since:
                self.loom_states[zone_id].state_duration_seconds = (timestamp - self.loom_states[zone_id].state_since).total_seconds()
            if current_state == "UNATTENDED":
                s = self.loom_states[zone_id]
                if s.absence_start_time is None and s.last_seen_time:
                    s.absence_start_time = s.last_seen_time
            elif current_state == "ATTENDED":
                self.loom_states[zone_id].absence_start_time = None
            return None

        if pending["target_state"] == target_state:
            pending["frames_count"] += 1
        else:
            pending["target_state"] = target_state
            pending["frames_count"] = 1

        if pending["frames_count"] >= 2:
            previous_state = current_state
            self.loom_states[zone_id].current_state = target_state
            self.loom_states[zone_id].state_since = timestamp
            self.loom_states[zone_id].state_duration_seconds = 0.0
            if target_state == "UNATTENDED" and self.loom_states[zone_id].last_seen_time:
                self.loom_states[zone_id].absence_start_time = self.loom_states[zone_id].last_seen_time
            elif target_state == "ATTENDED":
                self.loom_states[zone_id].absence_start_time = None
            pending["target_state"] = None
            pending["frames_count"] = 0
            print(f"[PersonNearMachine] Zone {previous_state} -> {target_state} @ {timestamp.isoformat()}")
            return previous_state
        return None

    def check_absence_alert(self, zone_id: str, zone_name: str, now: datetime) -> Optional[Dict[str, Any]]:
        if not self.config.notify_on_absence:
            return None
        state = self.loom_states.get(zone_id)
        if not state or state.current_state != "UNATTENDED" or state.absence_start_time is None:
            return None
        absence_duration_minutes = (now - state.absence_start_time).total_seconds() / 60.0
        if absence_duration_minutes < self.config.absence_threshold_minutes:
            return None
        if not hasattr(self, "last_absence_alert_time"):
            self.last_absence_alert_time = {}
        last_alert = self.last_absence_alert_time.get(zone_id)
        if last_alert and (now - last_alert).total_seconds() < self.config.alert_cooldown_minutes * 60.0:
            return None
        self.last_absence_alert_time[zone_id] = now
        return {
            "loom_id": zone_id,
            "loom_name": zone_name,
            "absence_duration_minutes": absence_duration_minutes,
            "absence_since": state.absence_start_time.isoformat() if state.absence_start_time else None,
        }

    def get_loom_state(self, zone_id: str) -> LoomPresenceState:
        return self.loom_states.get(zone_id)

    def get_all_states(self) -> Dict[str, LoomPresenceState]:
        return self.loom_states.copy()

    def cleanup_old_data(self, current_time: datetime) -> None:
        pass

    def reset(self) -> None:
        now = utc_now()
        for zone_id in self.loom_states:
            self.loom_states[zone_id].current_state = "UNKNOWN"
            self.loom_states[zone_id].state_since = None
            self._zone_initialized_at[zone_id] = now
            self.loom_states[zone_id].state_duration_seconds = 0.0
            self.loom_states[zone_id].confidence = 0.0
            self.loom_states[zone_id].last_seen_time = None
            self.loom_states[zone_id].absence_start_time = None
            self.presence_buffer[zone_id].clear()
            self.pending_transitions[zone_id] = {"target_state": None, "frames_count": 0}
