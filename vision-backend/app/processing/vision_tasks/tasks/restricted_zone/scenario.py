"""
Restricted zone scenario - Industrial Implementation
----------------------------------------------------

Industrial-grade restricted zone monitoring with:
- Box-polygon intersection detection (not just center point)
- Object tracking for stable IDs
- Multi-level alerts:
  * Level 1 (Orange): Person touches boundary
  * Level 2 (Red): Person fully inside zone
  * Level 3 (Frequent): Person inside >2 seconds
- Temporal stability (anti-flicker)
- Duration-based alerting
- Per-track state management
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioEvent,
    ScenarioFrameContext,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from app.processing.vision_tasks.tasks.tracking import SimpleTracker, Track

from .config import RestrictedZoneConfig
from .zone_utils import check_box_zone_intersection
from .state import RestrictedZoneState, TrackZoneState
from .report_storage import add_violation as report_add_violation, finalize_report as report_finalize


@register_scenario("restricted_zone")
class RestrictedZoneScenario(BaseScenario):
    """
    Industrial-grade restricted zone monitoring.
    
    Features:
    - Proper box-polygon intersection (not just center point)
    - Object tracking for stable person IDs
    - Multi-level alert system (touch/orange, inside/red, duration/frequent)
    - Instant trigger (no anti-flicker wait; alert as soon as person in zone)
    - Duration-based alerting (>2 seconds triggers frequent alerts)
    - Per-track state management
    """

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = RestrictedZoneConfig(config, pipeline_context.task)
        
        # Initialize tracker
        self.tracker = SimpleTracker(
            max_age=self.config_obj.tracker_max_age,
            min_hits=self.config_obj.tracker_min_hits,
            iou_threshold=self.config_obj.tracker_iou_threshold,
            score_threshold=self.config_obj.tracker_score_threshold,
            max_distance_threshold=self.config_obj.tracker_max_distance,
            max_distance_threshold_max=self.config_obj.tracker_max_distance_max,
            distance_growth_per_missed_frame=self.config_obj.tracker_distance_growth,
            use_kalman=self.config_obj.tracker_use_kalman,
        )
        
        # Initialize state manager
        self.zone_state = RestrictedZoneState()
        
        # Legacy state for compatibility
        self._state["last_alert_time"] = None
        self._state["objects_in_zone"] = False
        self._state["in_zone_indices"] = []  # Detection indices inside zone
        self._state["touch_indices"] = []
        self._state["inside_indices"] = []
        self._state["track_to_detection"] = {}
        # Report: single record per session (person count + duration)
        self._violation_track_ids: Set[int] = set()
        self._report_max_concurrent = 0

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process one frame with industrial-grade restricted zone detection.
        
        Flow:
        1. Filter detections by target class and confidence
        2. Update tracker to get stable track IDs
        3. Check box-zone intersection for each tracked person
        4. Update per-track state with stability confirmation
        5. Generate multi-level alerts based on state
        """
        # Step 1: Validate configuration
        if not self.config_obj.target_class or not self.config_obj.zone_coordinates:
            self._state["in_zone_indices"] = []
            self._state["touch_indices"] = []
            self._state["inside_indices"] = []
            return []
        
        frame = frame_context.frame
        frame_height, frame_width = frame.shape[:2]
        detections = frame_context.detections
        boxes = detections.boxes
        classes = detections.classes
        scores = detections.scores
        
        # Step 2: Filter detections by target class and confidence
        target_class = self.config_obj.target_class.lower()
        confidence_threshold = self.config_obj.confidence_threshold
        
        filtered_detections = []
        filtered_indices = []
        
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if isinstance(cls, str) and cls.lower() == target_class and score >= confidence_threshold:
                filtered_detections.append((box, score))
                filtered_indices.append(i)
        
        if len(filtered_detections) > 0:
            print(f"[RESTRICTED_ZONE] 📊 Frame: {len(filtered_detections)} person(s) detected "
                  f"(conf>={confidence_threshold:.2f})")
        
        # Step 3: Update tracker to get stable track IDs
        # Note: tracker.update returns confirmed active tracks
        tracked_objects = self.tracker.update(filtered_detections)
        
        if len(filtered_detections) > 0 and len(tracked_objects) == 0:
            print(f"[RESTRICTED_ZONE] ⚠️ No tracks yet (need {self.config_obj.tracker_min_hits} hits) | "
                  f"detections: {len(filtered_detections)}")
        
        # Step 4: Match tracks to detections and check zone intersection
        # We need to match tracks back to original detection indices
        track_to_detection = {}  # track_id -> detection_index
        touch_track_ids: Set[int] = set()
        inside_track_ids: Set[int] = set()
        
        # Match each track to its corresponding detection
        for track in tracked_objects:
            # Find best matching detection by comparing bbox
            best_match_idx = None
            best_match_distance = float('inf')
            
            for i, (bbox, _) in enumerate(filtered_detections):
                # Calculate center distance between track and detection
                track_center = ((track.bbox[0] + track.bbox[2]) / 2, (track.bbox[1] + track.bbox[3]) / 2)
                det_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                distance = ((track_center[0] - det_center[0]) ** 2 + (track_center[1] - det_center[1]) ** 2) ** 0.5
                
                if distance < best_match_distance and distance < 50.0:  # Max 50 pixels distance
                    best_match_distance = distance
                    best_match_idx = filtered_indices[i]
            
            if best_match_idx is None:
                continue
            
            track_to_detection[track.track_id] = best_match_idx

            # Capture state before update (to detect exit)
            prev_state = self.zone_state.track_states.get(track.track_id)
            entry_time_before = prev_state.entry_time if prev_state else None
            was_inside_before = bool(prev_state and prev_state.confirmed_inside)

            # Check box-zone intersection
            intersection_result = check_box_zone_intersection(
                track.bbox,
                self.config_obj.zone_coordinates,
                frame_width,
                frame_height,
            )
            touches = intersection_result['touches']
            inside = intersection_result['inside']
            
            # Debug: log first detection in zone
            if touches or inside:
                print(f"[RESTRICTED_ZONE] 🔍 Track {track.track_id}: touches={touches}, inside={inside}, "
                      f"ratio={intersection_result.get('intersection_ratio', 0):.2f}")

            # Update track state with stability confirmation
            track_state = self.zone_state.update_track_state(
                track.track_id,
                touches,
                inside,
                frame_context.timestamp,
                stability_frames=self.config_obj.stability_frames,
            )

            # Detect exit: was inside, now outside -> save violation to DB (single record)
            if was_inside_before and entry_time_before and not track_state.entry_time:
                exit_time = frame_context.timestamp
                duration_sec = (exit_time - entry_time_before).total_seconds()
                self._violation_track_ids.add(track.track_id)
                total_unique = len(self._violation_track_ids)
                report_add_violation(
                    agent_id=self.pipeline_context.agent_id,
                    agent_name=self.pipeline_context.agent_name,
                    camera_id=self.pipeline_context.camera_id or "",
                    track_id=track.track_id,
                    entry_time=entry_time_before,
                    exit_time=exit_time,
                    duration_seconds=duration_sec,
                    total_unique_persons=total_unique,
                    max_concurrent_in_zone=self._report_max_concurrent,
                )

            if track_state.confirmed_touches:
                touch_track_ids.add(track.track_id)
            if track_state.confirmed_inside:
                inside_track_ids.add(track.track_id)
        
        # Running max concurrent in zone (for report)
        self._report_max_concurrent = max(
            self._report_max_concurrent, len(inside_track_ids)
        )

        # Step 5: Update state for visualization (pipeline uses in_zone_indices to draw red boxes)
        self._state["track_to_detection"] = track_to_detection
        self._state["touch_indices"] = [
            track_to_detection[tid] for tid in touch_track_ids if tid in track_to_detection
        ]
        self._state["inside_indices"] = [
            track_to_detection[tid] for tid in inside_track_ids if tid in track_to_detection
        ]
        self._state["in_zone_indices"] = (
            self._state["touch_indices"] + self._state["inside_indices"]
        )
        # Log so you can see in terminal: color change = red/orange boxes
        if self._state["in_zone_indices"]:
            n_inside = len(self._state["inside_indices"])
            n_touch = len(self._state["touch_indices"])
            print(
                f"[RESTRICTED_ZONE] 🟥 {n_inside} person(s) in zone (red) | "
                f"🟧 {n_touch} touching (orange) | total red boxes: {len(self._state['in_zone_indices'])}"
            )

        # Step 6: Generate multi-level alerts
        events = self._generate_alerts(frame_context, tracked_objects, track_to_detection)
        
        # Step 7: Cleanup inactive tracks
        active_track_ids = {t.track_id for t in tracked_objects}
        self.zone_state.cleanup_inactive_tracks(active_track_ids)
        
        return events
    
    
    def _generate_alerts(
        self,
        frame_context: ScenarioFrameContext,
        tracked_objects: List[Track],
        track_to_detection: Dict[int, int],
    ) -> List[ScenarioEvent]:
        """
        Generate multi-level alerts based on track state.
        
        Alert Levels:
        - Level 1 (Touch/Orange): Person touches boundary
        - Level 2 (Inside/Red): Person fully inside zone
        - Level 3 (Duration/Frequent): Person inside >2 seconds (frequent alerts)
        """
        events = []
        now = frame_context.timestamp
        
        touch_track_ids = []
        inside_track_ids = []
        duration_violation_track_ids = []
        
        for track in tracked_objects:
            track_id = track.track_id
            track_state = self.zone_state.track_states.get(track_id)
            
            if not track_state:
                continue
            
            # Level 1: Touch alert (orange)
            if track_state.confirmed_touches and not track_state.confirmed_inside:
                touch_track_ids.append(track_id)
            
            # Level 2: Inside alert (red)
            if track_state.confirmed_inside:
                inside_track_ids.append(track_id)
                
                # Level 3: Duration violation (>2 seconds)
                duration = self.zone_state.get_duration_inside(track_id, now)
                if duration >= self.config_obj.duration_threshold_seconds:
                    duration_violation_track_ids.append(track_id)
        
        # Generate Level 1 alerts (Touch - Orange)
        if touch_track_ids:
            detection_indices = [
                track_to_detection[tid] for tid in touch_track_ids if tid in track_to_detection
            ]
            if detection_indices:
                # Check cooldown per track
                for track_id in touch_track_ids:
                    track_state = self.zone_state.track_states.get(track_id)
                    if track_state and self._should_alert(track_state, now, self.config_obj.alert_cooldown_seconds):
                        label_touch = self._generate_touch_label(len(touch_track_ids))
                        print(f"[RESTRICTED_ZONE] 🚨 Alert (orange): {label_touch}")
                        events.append(
                            ScenarioEvent(
                                event_type="restricted_zone_touch",
                                label=label_touch,
                                confidence=1.0,
                                metadata={
                                    "target_class": self.config_obj.target_class,
                                    "alert_level": "touch",
                                    "alert_color": "orange",
                                    "track_ids": touch_track_ids,
                                    "objects_count": len(touch_track_ids),
                                },
                                detection_indices=detection_indices,
                                timestamp=now,
                                frame_index=frame_context.frame_index,
                            )
                        )
                        track_state.last_alert_time = now
                        track_state.alert_level = "touch"
                        break  # One alert per frame for touch
        
        # Generate Level 2 alerts (Inside - Red)
        if inside_track_ids:
            detection_indices = [
                track_to_detection[tid] for tid in inside_track_ids if tid in track_to_detection
            ]
            if detection_indices:
                # Check cooldown per track
                for track_id in inside_track_ids:
                    track_state = self.zone_state.track_states.get(track_id)
                    if track_state and self._should_alert(track_state, now, self.config_obj.alert_cooldown_seconds):
                        label_inside = self._generate_inside_label(len(inside_track_ids))
                        print(f"[RESTRICTED_ZONE] 🚨 Alert (red): {label_inside}")
                        events.append(
                            ScenarioEvent(
                                event_type="restricted_zone_detection",
                                label=label_inside,
                                confidence=1.0,
                                metadata={
                                    "target_class": self.config_obj.target_class,
                                    "alert_level": "inside",
                                    "alert_color": "red",
                                    "track_ids": inside_track_ids,
                                    "objects_count": len(inside_track_ids),
                                },
                                detection_indices=detection_indices,
                                timestamp=now,
                                frame_index=frame_context.frame_index,
                            )
                        )
                        track_state.last_alert_time = now
                        track_state.alert_level = "inside"
                        break  # One alert per frame for inside
        
        # Generate Level 3 alerts (Duration Violation - Frequent)
        if duration_violation_track_ids:
            detection_indices = [
                track_to_detection[tid] for tid in duration_violation_track_ids if tid in track_to_detection
            ]
            if detection_indices:
                # Frequent alerts (shorter cooldown)
                for track_id in duration_violation_track_ids:
                    track_state = self.zone_state.track_states.get(track_id)
                    duration = self.zone_state.get_duration_inside(track_id, now)
                    
                    # Use shorter cooldown for duration violations
                    if track_state and self._should_alert(
                        track_state, now, self.config_obj.duration_alert_interval_seconds
                    ):
                        label_dur = self._generate_duration_label(duration, len(duration_violation_track_ids))
                        print(f"[RESTRICTED_ZONE] 🚨 Alert (duration): {label_dur}")
                        events.append(
                            ScenarioEvent(
                                event_type="restricted_zone_duration_violation",
                                label=label_dur,
                                confidence=1.0,
                                metadata={
                                    "target_class": self.config_obj.target_class,
                                    "alert_level": "duration",
                                    "alert_color": "red",
                                    "duration_seconds": duration,
                                    "track_ids": duration_violation_track_ids,
                                    "objects_count": len(duration_violation_track_ids),
                                },
                                detection_indices=detection_indices,
                                timestamp=now,
                                frame_index=frame_context.frame_index,
                            )
                        )
                        track_state.last_alert_time = now
                        track_state.alert_level = "duration"
                        break  # One alert per frame for duration
        
        return events
    
    def _should_alert(self, track_state: TrackZoneState, now: datetime, cooldown_seconds: float) -> bool:
        """Check if alert should be triggered (cooldown check)."""
        if track_state.last_alert_time is None:
            return True
        elapsed = (now - track_state.last_alert_time).total_seconds()
        return elapsed >= cooldown_seconds
    
    def _generate_touch_label(self, count: int) -> str:
        """Generate label for touch alert (Level 1 - Orange)."""
        custom_label = self.config_obj.custom_label
        if custom_label and isinstance(custom_label, str) and custom_label.strip():
            return f"{custom_label.strip()} - Near restricted zone"
        target_class = self.config_obj.target_class.replace("_", " ").title()
        if count == 1:
            return f"{target_class} near restricted zone"
        return f"{count} {target_class}(s) near restricted zone"
    
    def _generate_inside_label(self, count: int) -> str:
        """Generate label for inside alert (Level 2 - Red)."""
        custom_label = self.config_obj.custom_label
        if custom_label and isinstance(custom_label, str) and custom_label.strip():
            return f"{custom_label.strip()} - In restricted zone"
        target_class = self.config_obj.target_class.replace("_", " ").title()
        if count == 1:
            return f"{target_class} detected in restricted zone"
        return f"{count} {target_class}(s) detected in restricted zone"
    
    def _generate_duration_label(self, duration: float, count: int) -> str:
        """Generate label for duration violation alert (Level 3 - Frequent)."""
        custom_label = self.config_obj.custom_label
        duration_str = f"{duration:.1f}"
        if custom_label and isinstance(custom_label, str) and custom_label.strip():
            return f"{custom_label.strip()} - In restricted zone for {duration_str}s"
        target_class = self.config_obj.target_class.replace("_", " ").title()
        if count == 1:
            return f"{target_class} in restricted zone for {duration_str}s"
        return f"{count} {target_class}(s) in restricted zone for {duration_str}s"
    
    def generate_label(self, count: int) -> str:
        """Legacy method for backward compatibility."""
        return self._generate_inside_label(count)
    
    def reset(self) -> None:
        """Clear zone state, finalize report (single record in DB), reset tracker."""
        report_finalize(
            self.pipeline_context.agent_id,
            self.pipeline_context.camera_id or "",
        )
        self._state["last_alert_time"] = None
        self._state["objects_in_zone"] = False
        self._state["in_zone_indices"] = []
        self._state["touch_indices"] = []
        self._state["inside_indices"] = []
        self._state["track_to_detection"] = {}
        self._violation_track_ids.clear()
        self._report_max_concurrent = 0
        self.zone_state.reset()
