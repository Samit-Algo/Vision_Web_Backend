"""
Box Count Scenario
------------------

Counts boxes (or specified class) with two modes:
1. Simple per-frame counting (no zone)
2. Line-based counting with tracking (boxes crossing a line)

Same as ClassCountScenario but defaults to "box" class.
"""

from typing import List, Dict, Any, Optional

from app.processing.scenarios.contracts import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent
)
from app.processing.scenarios.registry import register_scenario
from app.processing.scenarios.box_count.config import BoxCountConfig
from app.processing.scenarios.class_count.counter import (
    count_class_detections,
    generate_count_label,
    filter_detections_by_class
)
from app.processing.scenarios.class_count.reporter import generate_report
from app.processing.scenarios.tracking import SimpleTracker, LineCrossingCounter


@register_scenario("box_count")
class BoxCountScenario(BaseScenario):
    """
    Counts boxes with optional line crossing.
    
    Same as class_count but defaults to "box" class.
    """
    
    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        
        # Load configuration
        self.config_obj = BoxCountConfig(config, pipeline_context.task)
        
        # Initialize tracker and line counter for line-based counting
        self.tracker: Optional[SimpleTracker] = None
        self.line_counter: Optional[LineCrossingCounter] = None
        
        if self.config_obj.zone_type == "line" and self.config_obj.zone_coordinates:
            self.tracker = SimpleTracker(
                max_age=self.config_obj.max_age,
                min_hits=self.config_obj.min_hits,
                iou_threshold=self.config_obj.iou_threshold,
                score_threshold=self.config_obj.score_threshold
            )
            self.line_counter = LineCrossingCounter(
                line_coordinates=self.config_obj.zone_coordinates,
                direction=self.config_obj.zone_direction
            )
    
    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process a single frame.
        
        Handles two modes:
        1. Simple counting (no zone)
        2. Line-based counting with tracking (line zone)
        """
        # Early exit: No target class configured
        if not self.config_obj.target_class:
            return []
        
        # Line-based counting (with tracking)
        if self.config_obj.zone_type == "line" and self.tracker and self.line_counter:
            return self._process_line_counting(frame_context)
        
        # Simple per-frame counting (no zone)
        return self._process_simple_counting(frame_context)
    
    def _process_line_counting(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """Process line-based counting with tracking."""
        # Filter detections by target class
        class_detections = filter_detections_by_class(
            frame_context.detections,
            self.config_obj.target_class
        )
        
        # Update tracker
        active_tracks = self.tracker.update(class_detections)
        
        # Check for line crossings
        for track in active_tracks:
            crossing = self.line_counter.check_crossing(track)
            if crossing:
                # Store crossing event in state
                if "crossing_events" not in self._state:
                    self._state["crossing_events"] = []
                self._state["crossing_events"].append({
                    "track_id": track.track_id,
                    "event": crossing,
                    "timestamp": frame_context.timestamp.isoformat()
                })
        
        # Get counts
        counts = self.line_counter.get_counts()
        
        # Determine which count to display based on direction
        if self.config_obj.zone_direction == "entry":
            count = counts["entry_count"]
        elif self.config_obj.zone_direction == "exit":
            count = counts["exit_count"]
        else:  # both
            count = counts["net_count"]
        
        # Generate label
        label = self._generate_line_count_label(counts)
        
        # Generate report
        report = generate_report(
            self._state,
            count,
            frame_context.timestamp,
            self.config_obj.zone_applied
        )
        report["line_counts"] = counts
        report["active_tracks"] = len(active_tracks)
        
        # Match tracks to detection indices for visualization
        matched_indices = self._match_tracks_to_detections(
            active_tracks,
            frame_context.detections
        )
        
        # Emit event
        event = ScenarioEvent(
            event_type="box_count",
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
        
        return [event]
    
    def _process_simple_counting(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """Process simple per-frame counting (no zone)."""
        # Count detections
        matched_count, matched_indices = count_class_detections(
            frame_context.detections,
            self.config_obj.target_class
        )
        
        # Generate report
        report = generate_report(
            self._state,
            matched_count,
            frame_context.timestamp,
            self.config_obj.zone_applied
        )
        
        # Generate label
        label = generate_count_label(
            matched_count,
            self.config_obj.target_class,
            self.config_obj.custom_label
        )
        
        # Emit event
        event = ScenarioEvent(
            event_type="box_count",
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
        
        return [event]
    
    def _generate_line_count_label(self, counts: Dict[str, int]) -> str:
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
            else:  # exit
                return f"{custom_label.strip()}: {exit_count}"
        else:
            if direction == "both":
                return f"{target_class} count: {net_count} (IN: {entry_count}, OUT: {exit_count})"
            elif direction == "entry":
                return f"{entry_count} {target_class}(s) crossed in"
            else:  # exit
                return f"{exit_count} {target_class}(s) crossed out"
    
    def _match_tracks_to_detections(self, tracks: List, detections) -> List[int]:
        """Match active tracks to detection indices for visualization."""
        matched_indices = []
        boxes = detections.boxes
        classes = detections.classes
        target_class = self.config_obj.target_class.lower()
        
        for track in tracks:
            track_bbox = track.bbox
            # Find matching detection by IoU overlap
            for idx, (det_box, det_class) in enumerate(zip(boxes, classes)):
                if isinstance(det_class, str) and det_class.lower() == target_class:
                    # Simple overlap check
                    if self._boxes_overlap(track_bbox, det_box):
                        matched_indices.append(idx)
                        break
        
        return matched_indices
    
    def _boxes_overlap(self, box1: List[float], box2: List[float]) -> bool:
        """Check if two boxes overlap (simple IoU > 0.3)."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        overlap_left = max(x1_1, x1_2)
        overlap_top = max(y1_1, y1_2)
        overlap_right = min(x2_1, x2_2)
        overlap_bottom = min(y2_1, y2_2)
        
        if overlap_right < overlap_left or overlap_bottom < overlap_top:
            return False
        
        intersection = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return False
        
        iou = intersection / union
        return iou >= 0.3
    
    def reset(self) -> None:
        """Reset scenario state."""
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
