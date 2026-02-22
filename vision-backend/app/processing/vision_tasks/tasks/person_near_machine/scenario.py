"""
Person Near Machine Monitoring Scenario (Industrial-Grade)
-----------------------------------------------------------
Polygon zone (like restricted_zone). Target class from agent config (e.g. person).
ATTENDED = someone in zone > min_presence_seconds. UNATTENDED = no one for > grace_time_seconds.
Zone color: green = attended, red = unattended, gray = unknown. Process at 5 FPS.

Code layout:
  - _point_in_polygon_pixel: helper to test if point is inside zone (normalized or pixel coords)
  - PersonNearMachineScenario: requires_yolo, __init__, process, update_internal_state, get_overlay_data, reset
"""

# -------- Imports --------
from typing import Any, Dict, List, Optional

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from app.processing.vision_tasks.tasks.class_count.counter import filter_detections_by_class_with_indices
from app.processing.vision_tasks.tasks.restricted_zone.zone_utils import point_in_polygon
from app.processing.vision_tasks.tracking import SimpleTracker
from .config import PersonNearMachineConfig
from .state import PersonNearMachineStateManager, ZONE_ID
from .types import PersonPresenceAnalysis


# ========== Helper: Point-in-polygon (normalized or pixel coordinates) ==========

def _point_in_polygon_pixel(
    center_x: float, center_y: float,
    zone_coordinates: List[List[float]],
    frame_width: int, frame_height: int,
) -> bool:
    """True if (center_x, center_y) is inside polygon. Handles normalized (0-1) or pixel coords."""
    if not zone_coordinates or len(zone_coordinates) < 3:
        return False
    max_val = max(max(p[0], p[1]) for p in zone_coordinates)
    if max_val <= 1.0 and frame_width and frame_height:
        # Zone is normalized; convert point to normalized
        nx = center_x / frame_width
        ny = center_y / frame_height
        return point_in_polygon((nx, ny), zone_coordinates)
    return point_in_polygon((center_x, center_y), zone_coordinates)


# ========== Scenario: Person Near Machine (zone attended / unattended) ==========

@register_scenario("person_near_machine")
class PersonNearMachineScenario(BaseScenario):
    """
    Person Near Machine: polygon zone (like restricted_zone), target-class-only detection (class from agent config).
    Green = attended, Red = unattended.
    """

    def requires_yolo_detections(self) -> bool:
        return True

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        task = getattr(pipeline_context, "task", None) or {}
        self.config_obj = PersonNearMachineConfig(config, task)
        # Require polygon zone (at least 3 points)
        if not self.config_obj.zone_coordinates or len(self.config_obj.zone_coordinates) < 3:
            raise ValueError("person_near_machine requires zone with type 'polygon' and at least 3 coordinates")
        fps = getattr(pipeline_context, "fps", None) or self.config_obj.fps
        fps = max(1, fps)
        self.state_manager = PersonNearMachineStateManager(self.config_obj, fps)
        self.tracker = SimpleTracker(
            max_age=self.config_obj.tracker_max_age,
            min_hits=self.config_obj.tracker_min_hits,
            iou_threshold=self.config_obj.tracker_iou_threshold,
            score_threshold=self.config_obj.confidence_threshold,
        )
        self.previous_state = "UNKNOWN"
        print(
            f"[PersonNearMachine] target_class={self.config_obj.target_class}, polygon zone ({len(self.config_obj.zone_coordinates)} points), "
            f"absence_threshold={self.config_obj.absence_threshold_minutes}min, "
            f"grace_time={self.config_obj.grace_time_seconds}s, min_presence={self.config_obj.min_presence_seconds}s, fps={fps}"
        )

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        events = []
        frame = frame_context.frame
        frame_index = frame_context.frame_index
        timestamp = frame_context.timestamp
        detections = frame_context.detections
        h, w = frame.shape[:2]
        # --- Filter by target class and update tracker ---
        target_class = self.config_obj.target_class
        person_detections, person_indices = filter_detections_by_class_with_indices(detections, target_class)
        tracker_detections = [(bbox, score) for bbox, score in person_detections]
        tracked_persons = self.tracker.update(tracker_detections)
        # --- Check which tracked persons are inside the polygon zone ---
        person_in_zone = False
        person_count = 0
        max_confidence = 0.0
        for track in tracked_persons:
            cx, cy = track.center
            if _point_in_polygon_pixel(cx, cy, self.config_obj.zone_coordinates, w, h):
                person_in_zone = True
                person_count += 1
                max_confidence = max(max_confidence, track.score)

        # --- Update state (ATTENDED / UNATTENDED) and emit transition event if enabled ---
        analysis = PersonPresenceAnalysis(
            loom_id=ZONE_ID,
            person_detected=person_in_zone,
            person_count=person_count,
            timestamp=timestamp,
            frame_index=frame_index,
            confidence=max_confidence if person_in_zone else 0.0,
        )
        previous_state = self.state_manager.add_presence_analysis(analysis)
        zone_state = self.state_manager.get_loom_state(ZONE_ID)
        current_state = zone_state.current_state if zone_state else "UNATTENDED"
        if self.config_obj.emit_state_transitions and previous_state and previous_state != current_state:
            events.append(
                ScenarioEvent(
                    event_type="person_presence_transition",
                    label=f"Zone changed from {previous_state} to {current_state}",
                    confidence=zone_state.confidence,
                    metadata={
                        "previous_state": previous_state,
                        "current_state": current_state,
                        "state_duration_seconds": zone_state.state_duration_seconds,
                        "person_count": person_count,
                        "transition_timestamp": timestamp.isoformat(),
                    },
                    detection_indices=person_indices if person_in_zone else [],
                    timestamp=timestamp,
                    frame_index=frame_index,
                )
            )
            print(f"[PersonNearMachineScenario] {previous_state} -> {current_state}")
        # --- Emit unattended alert if zone has been empty too long ---
        absence_alert = self.state_manager.check_absence_alert(ZONE_ID, "Zone", timestamp)
        if absence_alert:
            events.append(
                ScenarioEvent(
                    event_type="loom_unattended_alert",
                    label=f"Zone has been unattended for {absence_alert['absence_duration_minutes']:.1f} minutes",
                    confidence=1.0,
                    metadata={
                        "absence_duration_minutes": absence_alert["absence_duration_minutes"],
                        "absence_since": absence_alert.get("absence_since"),
                        "report": {"unattended_minutes": round(absence_alert["absence_duration_minutes"], 1)},
                    },
                    detection_indices=[],
                    timestamp=timestamp,
                    frame_index=frame_index,
                )
            )
            print(f"[PersonNearMachineScenario] Unattended alert: {absence_alert['absence_duration_minutes']:.1f} min")

        self.previous_state = current_state
        # --- Optional: emit periodic presence update (e.g. every 10 seconds at 5 FPS) ---
        if self.config_obj.emit_periodic_updates and frame_index % (getattr(self.pipeline_context, "fps", 5) * 10) == 0:
            events.append(
                ScenarioEvent(
                    event_type="person_presence_update",
                    label=f"Zone ({current_state})",
                    confidence=zone_state.confidence,
                    metadata={"state": current_state, "state_duration_seconds": zone_state.state_duration_seconds, "person_count": person_count},
                    detection_indices=person_indices if person_in_zone else [],
                    timestamp=timestamp,
                    frame_index=frame_index,
                )
            )

        self.state_manager.cleanup_old_data(timestamp)
        self.update_internal_state()  # Expose state for pipeline (zone color, etc.)
        return events

    # --------- Expose state for pipeline / UI (zone state, config) ---------
    def update_internal_state(self) -> None:
        all_states = self.state_manager.get_all_states()
        state = all_states.get(ZONE_ID)
        self._state = {
            "zone": {
                "current_state": state.current_state if state else "UNATTENDED",
                "state_since": state.state_since.isoformat() if state and state.state_since else None,
                "state_duration_seconds": state.state_duration_seconds if state else 0.0,
                "confidence": state.confidence if state else 0.0,
                "last_seen_time": state.last_seen_time.isoformat() if state and state.last_seen_time else None,
                "absence_start_time": state.absence_start_time.isoformat() if state and state.absence_start_time else None,
                "last_updated": state.last_updated.isoformat() if state else None,
            },
            "config": {
                "absence_threshold_minutes": self.config_obj.absence_threshold_minutes,
                "grace_time_seconds": self.config_obj.grace_time_seconds,
                "min_presence_seconds": self.config_obj.min_presence_seconds,
            },
        }

    # --------- Overlay: polygon + color for stream/UI (green/red/gray) ---------
    def get_overlay_data(self, frame_context=None) -> Optional[Dict[str, Any]]:
        """Return polygon and color: gray=UNKNOWN, green=ATTENDED, red=UNATTENDED."""
        all_states = self.state_manager.get_all_states()
        state = all_states.get(ZONE_ID)
        current_state = state.current_state if state else "UNKNOWN"
        if current_state == "ATTENDED":
            color = "#00ff00"  # Green
        elif current_state == "UNATTENDED":
            color = "#ff0000"  # Red
        else:
            color = "#808080"  # Gray for UNKNOWN (initial)
        return {
            "polygons": [self.config_obj.zone_coordinates],
            "colors": [color],
        }

    def reset(self) -> None:
        self.state_manager.reset()
        self.previous_state = "UNKNOWN"
        self._state.clear()
        self.tracker = SimpleTracker(
            max_age=self.config_obj.tracker_max_age,
            min_hits=self.config_obj.tracker_min_hits,
            iou_threshold=self.config_obj.tracker_iou_threshold,
            score_threshold=self.config_obj.confidence_threshold,
        )
