"""
Class Count Scenario
--------------------

Counts detections of a specified class with two modes:
1. Simple per-frame counting (no zone)
2. Line-based counting with tracking (objects touching a line)
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from app.processing.scenarios.contracts import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent
)
from app.processing.scenarios.registry import register_scenario
from app.processing.scenarios.class_count.config import ClassCountConfig
from app.processing.scenarios.class_count.counter import (
    count_class_detections,
    generate_count_label,
    filter_detections_by_class
)
from app.processing.scenarios.class_count.reporter import generate_report
from app.processing.scenarios.class_count.report_storage import (
    save_counting_event,
    initialize_report_session,
    finalize_report_session
)
from app.processing.scenarios.tracking import SimpleTracker, LineCrossingCounter


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
        
        # Load configuration
        self.config_obj = ClassCountConfig(config, pipeline_context.task)
        
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
        
        # Initialize report session when agent starts
        # This creates a summary document in MongoDB to track the entire session
        self._report_initialized = False
        self._initialize_report_session()
    
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
        # Update line counter with current frame dimensions (for percentage-based coordinates)
        frame_height, frame_width = frame_context.frame.shape[:2]
        self.line_counter.update_frame_dimensions(frame_width, frame_height)
        
        # Filter detections by target class
        class_detections = filter_detections_by_class(
            frame_context.detections,
            self.config_obj.target_class
        )
        
        # Debug: Log detection count
        if len(class_detections) > 0:
            print(f"[CLASS_COUNT] 📦 Filtered {len(class_detections)} '{self.config_obj.target_class}' detections for tracking")
        
        # Update tracker
        active_tracks = self.tracker.update(class_detections)
        
        # Also get all active tracks (including unconfirmed) for line crossing detection
        # This ensures we detect crossings even for tracks that haven't been confirmed yet
        all_active_tracks = self.tracker.get_all_active_tracks()
        
        # Debug: Log tracking status
        if len(all_active_tracks) > 0:
            print(f"[CLASS_COUNT] 🎯 Tracking {len(active_tracks)} confirmed {self.config_obj.target_class} tracks (total active: {len(all_active_tracks)})")
            # Log current counts
            counts = self.line_counter.get_counts()
            print(f"[CLASS_COUNT] 📊 Current counts - IN: {counts.get('entry_count', 0)}, OUT: {counts.get('exit_count', 0)}, NET: {counts.get('net_count', 0)}")
        
        # Check for line touches using ALL active tracks (not just confirmed)
        # This uses touch detection: count when center point touches the line
        touched_track_ids = []  # Track IDs that touched the line in this frame
        for track in all_active_tracks:
            # Use touch-based counting instead of crossing
            touch_result = self.line_counter.check_touch(track)
            if touch_result:
                touched_track_ids.append(track.track_id)
                
                # Print alert with ===================
                print("=" * 50)
                if touch_result == 'entry':
                    print(f"[CLASS_COUNT] ✅ {self.config_obj.target_class.capitalize()} TOUCHED LINE (ENTRY) - Track ID: {track.track_id}, Total IN: {self.line_counter.entry_count}")
                    print(f"[CLASS_COUNT]   Center point: ({track.center[0]:.1f}, {track.center[1]:.1f})")
                elif touch_result == 'exit':
                    print(f"[CLASS_COUNT] ✅ {self.config_obj.target_class.capitalize()} TOUCHED LINE (EXIT) - Track ID: {track.track_id}, Total OUT: {self.line_counter.exit_count}")
                    print(f"[CLASS_COUNT]   Center point: ({track.center[0]:.1f}, {track.center[1]:.1f})")
                print("=" * 50)
                
                # Store touch event in state
                if "touch_events" not in self._state:
                    self._state["touch_events"] = []
                self._state["touch_events"].append({
                    "track_id": track.track_id,
                    "event": touch_result,
                    "timestamp": frame_context.timestamp.isoformat(),
                    "frame_index": frame_context.frame_index
                })
                
                # Save counting event to MongoDB report
                self._save_counting_event_to_db(
                    track_id=track.track_id,
                    event_type=touch_result,  # "entry" or "exit"
                    timestamp=frame_context.timestamp,
                    active_tracks=len(all_active_tracks)
                )
        
        # Store touched track IDs for this frame (for visualization)
        self._state["current_frame_touched_tracks"] = touched_track_ids
        
        # Store track information for visualization (center points, track IDs)
        # This allows the frontend to draw center points in green and change box color to yellow when touching
        track_info = []
        for track in all_active_tracks:
            # Check if center point is currently touching the line
            touching_line = False
            if self.line_counter:
                touching_line = self.line_counter.is_track_touching(track)
            
            # Check if this track has been counted (touched the line)
            is_counted = False
            touch_info = None
            if self.line_counter:
                touch_info = self.line_counter.get_track_touch_info(track.track_id)
                if touch_info:
                    is_counted = touch_info.get('counted', False)
            
            track_info.append({
                "track_id": track.track_id,
                "center": [track.center[0], track.center[1]],  # Center point for green dot visualization
                "bbox": track.bbox,
                "touching_line": touching_line,  # True if currently touching line (for yellow box color)
                "counted": is_counted,  # True if this track has been counted
                "direction": touch_info.get('direction') if touch_info else None  # 'entry' or 'exit'
            })
        self._state["track_info"] = track_info
        
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
        report["all_active_tracks"] = len(all_active_tracks)
        
        # Match tracks to detection indices for visualization (use confirmed tracks)
        matched_indices = self._match_tracks_to_detections(
            active_tracks,
            frame_context.detections
        )
        
        # Emit event
        event = ScenarioEvent(
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
                return f"{entry_count} {target_class}(s) entered"
            else:  # exit
                return f"{exit_count} {target_class}(s) exited"
    
    def _match_tracks_to_detections(self, tracks: List, detections) -> List[int]:
        """Match active tracks to detection indices for visualization."""
        matched_indices = []
        boxes = detections.boxes
        classes = detections.classes
        target_class = self.config_obj.target_class.lower()
        
        for track in tracks:
            track_bbox = track.bbox
            best_iou = 0.0
            best_idx = -1
            
            # Find matching detection by IoU overlap
            for idx, (det_box, det_class) in enumerate(zip(boxes, classes)):
                if isinstance(det_class, str) and det_class.lower() == target_class:
                    # Calculate IoU
                    iou = self._calculate_iou(track_bbox, det_box)
                    if iou > best_iou and iou >= 0.3:
                        best_iou = iou
                        best_idx = idx
            
            if best_idx >= 0 and best_idx not in matched_indices:
                matched_indices.append(best_idx)
        
        return matched_indices
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two boxes."""
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
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _initialize_report_session(self) -> None:
        """
        Initialize report session in MongoDB when agent starts.
        
        This is called once when the scenario is created (agent starts).
        Creates a summary document to track the entire counting session.
        """
        if self._report_initialized:
            return  # Already initialized
        
        try:
            # Get agent information from pipeline context
            agent_id = self.pipeline_context.agent_id
            agent_name = self.pipeline_context.agent_name
            camera_id = self.pipeline_context.camera_id or ""
            
            # Initialize report session in MongoDB
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
            # Don't crash if report initialization fails
            print(f"[CLASS_COUNT] ⚠️ Failed to initialize report session: {e}")
    
    def _save_counting_event_to_db(
        self,
        track_id: int,
        event_type: str,  # "entry" or "exit"
        timestamp: datetime,
        active_tracks: int
    ) -> None:
        """
        Save a single touch event to MongoDB.
        
        This is called every time an object's center point touches the counting line.
        Stores the event immediately to MongoDB.
        
        Args:
            track_id: Unique track ID of the object
            event_type: "entry" or "exit" (based on which side object came from)
            timestamp: When the event happened
            active_tracks: Number of objects currently being tracked
        """
        try:
            # Get agent information from pipeline context
            agent_id = self.pipeline_context.agent_id
            agent_name = self.pipeline_context.agent_name
            camera_id = self.pipeline_context.camera_id or ""
            
            # Get current counts from line counter
            if self.line_counter:
                counts = self.line_counter.get_counts()
                entry_count = counts["entry_count"]
                exit_count = counts["exit_count"]
            else:
                entry_count = 0
                exit_count = 0
            
            # Save event to MongoDB
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
            # Don't crash if saving fails
            print(f"[CLASS_COUNT] ⚠️ Failed to save counting event: {e}")
    
    def _finalize_report_session(self) -> None:
        """
        Finalize report session in MongoDB when agent stops.
        
        This is called when the scenario is reset (agent stops).
        Updates the summary document with final counts and end time.
        """
        if not self._report_initialized:
            return  # Never initialized, nothing to finalize
        
        try:
            # Get agent information
            agent_id = self.pipeline_context.agent_id
            
            # Get final counts
            if self.line_counter:
                counts = self.line_counter.get_counts()
                final_entry_count = counts["entry_count"]
                final_exit_count = counts["exit_count"]
                # Get active tracks as final standby count
                if self.tracker:
                    all_active_tracks = self.tracker.get_all_active_tracks()
                    final_standby_count = len(all_active_tracks)
                else:
                    final_standby_count = 0
            else:
                final_entry_count = 0
                final_exit_count = 0
                final_standby_count = 0
            
            # Finalize session in MongoDB
            finalize_report_session(
                agent_id=agent_id,
                report_type="class_count",
                final_entry_count=final_entry_count,
                final_exit_count=final_exit_count,
                final_standby_count=final_standby_count
            )
        except Exception as e:
            # Don't crash if finalization fails
            print(f"[CLASS_COUNT] ⚠️ Failed to finalize report session: {e}")
    
    def reset(self) -> None:
        """Reset scenario state."""
        # Finalize report session when agent stops
        # This updates the summary document with final counts
        self._finalize_report_session()
        
        # Clear state
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
        
        # Reset report initialization flag
        self._report_initialized = False
