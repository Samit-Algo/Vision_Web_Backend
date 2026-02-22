"""
Class Count Scenario
---------------------
Counts target class: (1) simple per-frame count (no zone), or (2) line-based entry/exit with tracker.
Uses counter, reporter, report_storage for DB. Pipeline uses track_info for drawing.

Code layout:
  - ClassCountScenario: __init__ (config, tracker, line_counter), process → process_line_counting | process_simple_counting,
    generate_line_count_label, match_tracks_to_detections, initialize_report_session, save_counting_event_to_db, finalize_report_session, reset.
"""

# -------- Imports --------
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent
)
from app.processing.vision_tasks.task_lookup import register_scenario
from .config import ClassCountConfig
from .counter import (
    count_class_detections,
    generate_count_label,
    filter_detections_by_class,
    calculate_iou,
)
from .reporter import generate_report
from .report_storage import (
    save_counting_event,
    initialize_report_session,
    finalize_report_session
)
from app.processing.vision_tasks.tracking import SimpleTracker, LineCrossingCounter


# ========== Scenario: Class count (simple or line-based) ==========

@register_scenario("class_count")
class ClassCountScenario(BaseScenario):
    """
    Counts class detections with optional line-based counting.

    Modes:
    - No zone: Simple per-frame count
    - Line zone: Counts objects when center point touches the line (with tracking)
    """

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = ClassCountConfig(config, pipeline_context.task)
        self.tracker: Optional[SimpleTracker] = None
        self.line_counter: Optional[LineCrossingCounter] = None

        if self.config_obj.zone_type == "line" and self.config_obj.zone_coordinates:
            self.tracker = SimpleTracker(
                max_age=self.config_obj.max_age,
                min_hits=self.config_obj.min_hits,
                iou_threshold=self.config_obj.iou_threshold,
                score_threshold=self.config_obj.score_threshold
            )
            # Same count_mode as box_count: entry_exit for separate IN/OUT counts and color by direction
            self.line_counter = LineCrossingCounter(
                line_coordinates=self.config_obj.zone_coordinates,
                direction=self.config_obj.zone_direction,
                count_mode="entry_exit",
            )
        self._report_initialized = False
        self.initialize_report_session()

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process a single frame.

        Handles two modes:
        1. Simple counting (no zone)
        2. Line-based counting with tracking (line zone)
        """
        if not self.config_obj.target_class:
            return []
        if self.config_obj.zone_type == "line" and self.tracker and self.line_counter:
            return self.process_line_counting(frame_context)

        # Simple per-frame counting (no zone)
        return self.process_simple_counting(frame_context)

    def process_line_counting(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """Line zone: filter by class → tracker update → line touch check → track_info + counts → event."""
        frame_height, frame_width = frame_context.frame.shape[:2]
        self.line_counter.update_frame_dimensions(frame_width, frame_height)
        class_detections = filter_detections_by_class(
            frame_context.detections,
            self.config_obj.target_class
        )

        active_tracks = self.tracker.update(class_detections)
        all_active_tracks = self.tracker.get_all_active_tracks()
        # --- Which tracks touched the line this frame; save events to DB ---
        touched_track_ids = []
        for track in all_active_tracks:
            touch_result = self.line_counter.check_touch(track)
            if touch_result:
                touched_track_ids.append(track.track_id)
                if "touch_events" not in self._state:
                    self._state["touch_events"] = []
                self._state["touch_events"].append({
                    "track_id": track.track_id,
                    "event": touch_result,
                    "timestamp": frame_context.timestamp.isoformat(),
                    "frame_index": frame_context.frame_index
                })
                self.save_counting_event_to_db(
                    track_id=track.track_id,
                    event_type=touch_result,
                    timestamp=frame_context.timestamp,
                    active_tracks=len(all_active_tracks)
                )

        self._state["current_frame_touched_tracks"] = touched_track_ids
        # --- Build track_info for pipeline (center, bbox, touching_line, counted, direction) ---
        track_info = []
        for track in all_active_tracks:
            touching_line = self.line_counter.is_track_touching(track) if self.line_counter else False
            touch_info = self.line_counter.get_track_touch_info(track.track_id) if self.line_counter else None
            is_counted = touch_info.get("counted", False) if touch_info else False
            track_info.append({
                "track_id": track.track_id,
                "center": [track.center[0], track.center[1]],
                "bbox": track.bbox,
                "touching_line": touching_line,
                "counted": is_counted,
                "direction": touch_info.get("direction") if touch_info else None
            })
        self._state["track_info"] = track_info

        counts = self.line_counter.get_counts()
        if self.config_obj.zone_direction == "entry":
            count = counts["entry_count"]
        elif self.config_obj.zone_direction == "exit":
            count = counts["exit_count"]
        else:
            count = counts["net_count"]

        label = self.generate_line_count_label(counts)
        report = generate_report(
            self._state,
            count,
            frame_context.timestamp,
            self.config_obj.zone_applied
        )
        report["line_counts"] = counts
        report["active_tracks"] = len(active_tracks)
        report["all_active_tracks"] = len(all_active_tracks)
        report["track_info"] = track_info
        report["entry_count"] = counts["entry_count"]
        report["exit_count"] = counts["exit_count"]

        matched_indices = self.match_tracks_to_detections(
            active_tracks,
            frame_context.detections
        )

        return [
            ScenarioEvent(
                event_type="class_count",
                label=label,
                confidence=1.0,
                metadata={
                    "count": count,
                    "entry_count": counts["entry_count"],
                    "exit_count": counts["exit_count"],
                    "net_count": counts["net_count"],
                    "target_class": self.config_obj.target_class,
                    "zone_type": "line",
                    "zone_direction": self.config_obj.zone_direction,
                    "active_tracks": len(active_tracks),
                    "report": report
                },
                detection_indices=matched_indices,
                timestamp=frame_context.timestamp,
                frame_index=frame_context.frame_index
            )
        ]

    def process_simple_counting(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """No zone: count detections of target_class this frame; return one event with count and report."""
        matched_count, matched_indices = count_class_detections(
            frame_context.detections,
            self.config_obj.target_class
        )

        report = generate_report(
            self._state,
            matched_count,
            frame_context.timestamp,
            self.config_obj.zone_applied
        )

        label = generate_count_label(
            matched_count,
            self.config_obj.target_class,
            self.config_obj.custom_label
        )

        return [
            ScenarioEvent(
                event_type="class_count",
                label=label,
                confidence=1.0,
                metadata={
                    "count": matched_count,
                    "target_class": self.config_obj.target_class,
                    "zone_type": "none",
                    "zone_applied": False,
                    "report": report
                },
                detection_indices=matched_indices,
                timestamp=frame_context.timestamp,
                frame_index=frame_context.frame_index
            )
        ]

    def generate_line_count_label(self, counts: Dict[str, int]) -> str:
        """Generate label for line-based counting."""
        custom_label = self.config_obj.custom_label
        direction = self.config_obj.zone_direction
        target_class = self.config_obj.target_class
        entry_count = counts["entry_count"]
        exit_count = counts["exit_count"]
        net_count = counts["net_count"]

        if custom_label and isinstance(custom_label, str) and custom_label.strip():
            if direction == "both":
                return f"{custom_label.strip()}: {net_count} (IN: {entry_count}, OUT: {exit_count})"
            elif direction == "entry":
                return f"{custom_label.strip()}: {entry_count}"
            else:
                return f"{custom_label.strip()}: {exit_count}"
        else:
            if direction == "both":
                return f"{target_class} count: {net_count} (IN: {entry_count}, OUT: {exit_count})"
            elif direction == "entry":
                return f"{entry_count} {target_class}(s) entered"
            else:
                return f"{exit_count} {target_class}(s) exited"

    def match_tracks_to_detections(self, tracks: List, detections) -> List[int]:
        """Match active tracks to detection indices for visualization."""
        matched_indices = []
        boxes = detections.boxes
        classes = detections.classes
        target_class = self.config_obj.target_class.lower()

        for track in tracks:
            track_bbox = track.bbox
            best_iou = 0.0
            best_idx = -1
            for idx, (det_box, det_class) in enumerate(zip(boxes, classes)):
                if isinstance(det_class, str) and det_class.lower() == target_class:
                    iou = calculate_iou(track_bbox, list(det_box))
                    if iou > best_iou and iou >= 0.3:
                        best_iou = iou
                        best_idx = idx
            if best_idx >= 0 and best_idx not in matched_indices:
                matched_indices.append(best_idx)

        return matched_indices

    def initialize_report_session(self) -> None:
        """Initialize report session in MongoDB when agent starts."""
        if self._report_initialized:
            return
        try:
            agent_id = self.pipeline_context.agent_id
            agent_name = self.pipeline_context.agent_name
            camera_id = self.pipeline_context.camera_id or ""
            success = initialize_report_session(
                agent_id=agent_id,
                agent_name=agent_name,
                camera_id=camera_id,
                report_type="class_count",
                target_class=self.config_obj.target_class
            )
            if success:
                self._report_initialized = True
        except Exception as e:
            print(f"[CLASS_COUNT] ⚠️ Failed to initialize report session: {e}")

    def save_counting_event_to_db(
        self,
        track_id: int,
        event_type: str,
        timestamp: datetime,
        active_tracks: int
    ) -> None:
        """Save a single touch event to MongoDB."""
        try:
            agent_id = self.pipeline_context.agent_id
            agent_name = self.pipeline_context.agent_name
            camera_id = self.pipeline_context.camera_id or ""
            if self.line_counter:
                counts = self.line_counter.get_counts()
                entry_count = counts["entry_count"]
                exit_count = counts["exit_count"]
            else:
                entry_count = 0
                exit_count = 0
            save_counting_event(
                agent_id=agent_id,
                agent_name=agent_name,
                camera_id=camera_id,
                report_type="class_count",
                track_id=track_id,
                event_type=event_type,
                timestamp=timestamp,
                entry_count=entry_count,
                exit_count=exit_count,
                active_tracks=active_tracks,
                target_class=self.config_obj.target_class
            )
        except Exception as e:
            print(f"[CLASS_COUNT] ⚠️ Failed to save counting event: {e}")

    def finalize_report_session(self) -> None:
        """Finalize report session in MongoDB when agent stops."""
        if not self._report_initialized:
            return
        try:
            agent_id = self.pipeline_context.agent_id
            if self.line_counter:
                counts = self.line_counter.get_counts()
                final_entry_count = counts["entry_count"]
                final_exit_count = counts["exit_count"]
                if self.tracker:
                    all_active_tracks = self.tracker.get_all_active_tracks()
                    final_standby_count = len(all_active_tracks)
                else:
                    final_standby_count = 0
            else:
                final_entry_count = 0
                final_exit_count = 0
                final_standby_count = 0
            finalize_report_session(
                agent_id=agent_id,
                report_type="class_count",
                final_entry_count=final_entry_count,
                final_exit_count=final_exit_count,
                final_standby_count=final_standby_count
            )
        except Exception as e:
            print(f"[CLASS_COUNT] ⚠️ Failed to finalize report session: {e}")

    def reset(self) -> None:
        """Reset scenario state."""
        self.finalize_report_session()
        self._state.clear()
        if self.tracker:
            self.tracker = SimpleTracker(
                max_age=self.config_obj.max_age,
                min_hits=self.config_obj.min_hits,
                iou_threshold=self.config_obj.iou_threshold,
                score_threshold=self.config_obj.score_threshold
            )
        if self.line_counter:
            self.line_counter.reset()
        self._report_initialized = False
