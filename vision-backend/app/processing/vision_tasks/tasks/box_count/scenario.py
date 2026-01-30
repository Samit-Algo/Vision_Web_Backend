"""
Box Count Scenario
------------------

Counts boxes (or specified class) with two modes:
1. Simple per-frame counting (no zone)
2. Line-based counting with tracking (boxes crossing a line)
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from app.processing.vision_tasks.tracking import SimpleTracker, LineCrossingCounter
from .config import BoxCountConfig
from app.processing.vision_tasks.tasks.class_count.counter import (
    count_class_detections,
    generate_count_label,
    filter_detections_by_class,
)
from app.processing.vision_tasks.tasks.class_count.reporter import generate_report
from app.processing.vision_tasks.tasks.class_count.report_storage import (
    save_counting_event,
    initialize_report_session,
    finalize_report_session,
)


@register_scenario("box_count")
class BoxCountScenario(BaseScenario):
    """
    Counts boxes with optional line crossing.
    Same as class_count but defaults to "box" class; supports line-based entry/exit counting.
    """

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = BoxCountConfig(config, pipeline_context.task)
        self.tracker: Optional[SimpleTracker] = None
        self.line_counter: Optional[LineCrossingCounter] = None
        if self.config_obj.zone_type == "line" and self.config_obj.zone_coordinates:
            self.tracker = SimpleTracker(
                max_age=self.config_obj.max_age,
                min_hits=self.config_obj.min_hits,
                iou_threshold=self.config_obj.iou_threshold,
                score_threshold=self.config_obj.score_threshold,
            )
            self.line_counter = LineCrossingCounter(
                line_coordinates=self.config_obj.zone_coordinates,
                direction=self.config_obj.zone_direction,
                count_mode="entry_exit",
            )
        self._report_initialized = False

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        if not self.config_obj.target_class:
            return []
        if self.config_obj.zone_type == "line" and self.tracker and self.line_counter:
            return self._process_line_counting(frame_context)
        return self._process_simple_counting(frame_context)

    def _process_line_counting(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        frame_height, frame_width = frame_context.frame.shape[:2]
        self.line_counter.update_frame_dimensions(frame_width, frame_height)
        class_detections = filter_detections_by_class(
            frame_context.detections, self.config_obj.target_class
        )
        active_tracks = self.tracker.update(class_detections)
        all_active_tracks = self.tracker.get_all_active_tracks()
        for track in all_active_tracks:
            touch_result = self.line_counter.check_touch(track)
            if touch_result:
                if "touch_events" not in self._state:
                    self._state["touch_events"] = []
                self._state["touch_events"].append({
                    "track_id": track.track_id,
                    "event": touch_result,
                    "timestamp": frame_context.timestamp.isoformat(),
                    "frame_index": frame_context.frame_index,
                })
                self._save_counting_event_to_db(
                    track_id=track.track_id,
                    event_type=touch_result,
                    timestamp=frame_context.timestamp,
                    active_tracks=len(all_active_tracks),
                )
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
                "direction": touch_info.get("direction") if touch_info else None,
            })
        self._state["track_info"] = track_info
        counts = self.line_counter.get_counts()
        if self.line_counter.count_mode == "single":
            count = counts["boxes_counted"]
        else:
            if self.config_obj.zone_direction == "entry":
                count = counts["entry_count"]
            elif self.config_obj.zone_direction == "exit":
                count = counts["exit_count"]
            else:
                count = counts["net_count"]
        label = self._generate_line_count_label(counts)
        report = generate_report(
            self._state, count, frame_context.timestamp, self.config_obj.zone_applied
        )
        report["line_counts"] = counts
        report["active_tracks"] = len(active_tracks)
        report["track_info"] = track_info
        report["entry_count"] = counts.get("entry_count", 0)
        report["exit_count"] = counts.get("exit_count", 0)
        matched_indices = self._match_tracks_to_detections(active_tracks, frame_context.detections)
        if not self._report_initialized and self.pipeline_context:
            ctx = self.pipeline_context
            initialize_report_session(
                agent_id=ctx.agent_id or "",
                agent_name=ctx.agent_name or "",
                camera_id=ctx.camera_id or "",
                report_type="box_count",
                target_class=self.config_obj.target_class or "box",
            )
            self._report_initialized = True
        metadata = {
            "count": count,
            "target_class": self.config_obj.target_class,
            "zone_type": "line",
            "zone_direction": self.config_obj.zone_direction,
            "active_tracks": len(active_tracks),
            "report": report,
            "entry_count": counts.get("entry_count", 0),
            "exit_count": counts.get("exit_count", 0),
            "net_count": counts.get("net_count", 0),
        }
        return [
            ScenarioEvent(
                event_type="box_count",
                label=label,
                confidence=1.0,
                metadata=metadata,
                detection_indices=matched_indices,
                timestamp=frame_context.timestamp,
                frame_index=frame_context.frame_index,
            )
        ]

    def _process_simple_counting(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        matched_count, matched_indices = count_class_detections(
            frame_context.detections,
            self.config_obj.target_class,
            self.config_obj.zone_coordinates,
        )
        report = generate_report(
            self._state, matched_count, frame_context.timestamp, self.config_obj.zone_applied
        )
        label = generate_count_label(
            matched_count, self.config_obj.target_class, self.config_obj.custom_label
        )
        return [
            ScenarioEvent(
                event_type="box_count",
                label=label,
                confidence=1.0,
                metadata={
                    "count": matched_count,
                    "target_class": self.config_obj.target_class,
                    "zone_type": "none",
                    "zone_applied": False,
                    "report": report,
                },
                detection_indices=matched_indices,
                timestamp=frame_context.timestamp,
                frame_index=frame_context.frame_index,
            )
        ]

    def _generate_line_count_label(self, counts: Dict[str, int]) -> str:
        if self.line_counter and self.line_counter.count_mode == "single":
            boxes_counted = counts.get("boxes_counted", 0)
            if self.config_obj.custom_label and isinstance(self.config_obj.custom_label, str) and self.config_obj.custom_label.strip():
                return f"{self.config_obj.custom_label.strip()}: {boxes_counted}"
            return f"BOXES COUNTED: {boxes_counted}"
        entry_count = counts.get("entry_count", 0)
        exit_count = counts.get("exit_count", 0)
        net_count = counts.get("net_count", 0)
        direction = self.config_obj.zone_direction
        target_class = self.config_obj.target_class
        if self.config_obj.custom_label and isinstance(self.config_obj.custom_label, str) and self.config_obj.custom_label.strip():
            if direction == "both":
                return f"{self.config_obj.custom_label.strip()}: {net_count} (IN: {entry_count}, OUT: {exit_count})"
            elif direction == "entry":
                return f"{self.config_obj.custom_label.strip()}: {entry_count}"
            else:
                return f"{self.config_obj.custom_label.strip()}: {exit_count}"
        if direction == "both":
            return f"{target_class} count: {net_count} (IN: {entry_count}, OUT: {exit_count})"
        elif direction == "entry":
            return f"{entry_count} {target_class}(s) crossed in"
        else:
            return f"{exit_count} {target_class}(s) crossed out"

    def _match_tracks_to_detections(self, tracks: List, detections) -> List[int]:
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
                    iou = self._calculate_iou(track_bbox, list(det_box))
                    if iou > best_iou and iou >= 0.3:
                        best_iou = iou
                        best_idx = idx
            if best_idx >= 0 and best_idx not in matched_indices:
                matched_indices.append(best_idx)
        return matched_indices

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        overlap_left = max(x1_1, x1_2)
        overlap_top = max(y1_1, y1_2)
        overlap_right = min(x2_1, x2_2)
        overlap_bottom = min(y2_1, y2_2)
        if overlap_right < overlap_left or overlap_bottom < overlap_top:
            return 0.0
        intersection = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return (intersection / union) if union else 0.0

    def _save_counting_event_to_db(
        self, track_id: int, event_type: str, timestamp: datetime, active_tracks: int
    ) -> None:
        try:
            ctx = self.pipeline_context
            if not ctx:
                return
            if self.line_counter:
                counts = self.line_counter.get_counts()
                entry_count = counts.get("entry_count", 0)
                exit_count = counts.get("exit_count", 0)
            else:
                entry_count = exit_count = 0
            save_counting_event(
                agent_id=ctx.agent_id or "",
                agent_name=ctx.agent_name or "",
                camera_id=ctx.camera_id or "",
                report_type="box_count",
                track_id=track_id,
                event_type=event_type,
                timestamp=timestamp,
                entry_count=entry_count,
                exit_count=exit_count,
                active_tracks=active_tracks,
                target_class=self.config_obj.target_class or "box",
            )
        except Exception as e:
            print(f"[BOX_COUNT] Failed to save counting event: {e}")

    def reset(self) -> None:
        if self._report_initialized and self.pipeline_context and self.line_counter:
            try:
                counts = self.line_counter.get_counts()
                final_entry = counts.get("entry_count", 0)
                final_exit = counts.get("exit_count", 0)
                final_standby = len(self.tracker.get_all_active_tracks()) if self.tracker else 0
                finalize_report_session(
                    self.pipeline_context.agent_id or "",
                    "box_count",
                    final_entry,
                    final_exit,
                    final_standby,
                )
            except Exception:
                pass
        self._state.clear()
        if self.config_obj.zone_type == "line" and self.config_obj.zone_coordinates:
            self.tracker = SimpleTracker(
                max_age=self.config_obj.max_age,
                min_hits=self.config_obj.min_hits,
                iou_threshold=self.config_obj.iou_threshold,
                score_threshold=self.config_obj.score_threshold,
            )
        if self.line_counter:
            self.line_counter.reset()
        self._report_initialized = False
