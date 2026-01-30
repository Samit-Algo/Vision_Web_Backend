"""
Pipeline Runner
---------------

Orchestrates the processing pipeline stages.
Manages the main processing loop, FPS pacing, and stage sequencing.
"""

import time
from typing import Optional, List, Any, Dict
from datetime import datetime

from app.processing.sources.contracts import FramePacket
from app.processing.sources.source_factory import Source
from app.processing.pipeline.context import PipelineContext
from app.processing.pipeline.stages import (
    acquire_frame_stage,
    inference_stage,
    merge_detections_stage,
    evaluate_rules_stage
)
from app.processing.pipeline.contracts import RuleMatch
from app.processing.models.contracts import Model
from app.utils.db import get_collection
from app.utils.datetime_utils import now, utc_now
from app.utils.event_notifier import send_event_to_backend_sync
from app.utils.event_session_manager import get_event_session_manager
from app.processing.worker.frame_processor import (
    draw_bounding_boxes, 
    draw_pose_keypoints,
    draw_box_count_annotations
)
from app.processing.detections.contracts import DetectionPacket


def format_video_time_ms(milliseconds: float) -> str:
    """Convert milliseconds to H:MM:SS.mmm string."""
    if milliseconds < 0:
        milliseconds = 0
    total_seconds, milliseconds_remainder = divmod(int(milliseconds), 1000)
    hours, remaining_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{milliseconds_remainder:03d}"


class PipelineRunner:
    """
    Orchestrates the processing pipeline.
    
    Manages the main loop, FPS pacing, and calls stages in sequence.
    Handles both continuous and patrol modes.
    """
    
    def __init__(
        self,
        context: PipelineContext,
        source: Source,
        models: List[Model],
        shared_store: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pipeline runner.
        
        Args:
            context: Pipeline context (task config and state)
            source: Source instance for frame acquisition
            models: List of loaded models
            shared_store: Optional shared memory dict for publishing frames
        """
        self.context = context
        self.source = source
        self.models = models
        self.shared_store = shared_store
        self.tasks_collection = get_collection()
    
    def run_continuous(self) -> None:
        """Run pipeline in continuous mode (process frames indefinitely at target FPS)."""
        # Check if any rules are counting rules (box_count or class_count)
        # For counting rules, use camera FPS instead of configured FPS
        use_camera_fps = False
        if self.context.rules:
            for rule in self.context.rules:
                rule_type = str(rule.get("type", "")).strip().lower()
                if rule_type in ['box_count', 'class_count']:
                    use_camera_fps = True
                    break
        
        # Default to configured FPS
        fps = self.context.fps
        min_interval = 1.0 / max(1, fps)
        next_tick = time.time()
        self.context.frame_index = 0
        last_status = time.time()
        processed_in_window = 0
        skipped_in_window = 0
        
        while True:
            # Check stop conditions
            stop_reason = self.context.get_stop_condition(self.tasks_collection)
            if stop_reason:
                status = "cancelled" if stop_reason == "stop_requested" else "completed"
                self.context.update_status(self.tasks_collection, status)
                print(f"[worker {self.context.task_id}] ⏹️ Stopping (reason={stop_reason})")
                return
            
            # Stage 1: Acquire frame
            frame_packet = acquire_frame_stage(self.source)
            if frame_packet is None:
                time.sleep(0.05)
                continue
            
            # For counting rules, use camera FPS if available
            if use_camera_fps and frame_packet.fps is not None and frame_packet.fps > 0:
                fps = frame_packet.fps
                min_interval = 1.0 / max(1, fps)
            
            # FPS pacing
            now_timestamp = time.time()
            if now_timestamp < next_tick:
                time.sleep(max(0.0, next_tick - now_timestamp))
            next_tick = max(next_tick + min_interval, time.time())
            
            # Update context tracking
            hub_index = frame_packet.frame_index
            if self.context.last_seen_hub_index is not None and hub_index > self.context.last_seen_hub_index + 1:
                skipped_in_window += (hub_index - self.context.last_seen_hub_index - 1)
            self.context.last_seen_hub_index = hub_index
            self.context.frame_index += 1
            processed_in_window += 1
            
            # Status logging (once per second)
            if (time.time() - last_status) >= 1.0:
                base_fps = frame_packet.fps if frame_packet.fps is not None else 0.0
                hub_index_status = self.context.last_seen_hub_index or 0
                print(f"[worker {self.context.task_id}] ⏱️ sampling agent_fps={fps} base_fps={base_fps} processed={processed_in_window}/s skipped={skipped_in_window} hub_frame_index={hub_index_status} camera_id={self.context.camera_id}")
                processed_in_window = 0
                skipped_in_window = 0
                last_status = time.time()
            
            # Stage 2: Inference
            detection_packets = inference_stage(self.context, frame_packet, self.models)
            
            # Stage 3: Merge detections
            merged_packet = merge_detections_stage(detection_packets)
            
            # Debug logging
            if self.context.frame_index == 1 or self.context.frame_index % 30 == 0:
                if merged_packet.keypoints:
                    print(f"[worker {self.context.task_id}] ✅ Keypoints extracted: {len(merged_packet.keypoints)} person(s) with pose data")
                else:
                    print(f"[worker {self.context.task_id}] ⚠️ No keypoints extracted! Check if model is a pose model (e.g., yolov8n-pose.pt)")
            
            # Stage 4: Evaluate rules (uses rules field from database)
            # Rules can use traditional handlers OR scenario implementations internally
            rule_match = None
            all_matched_indices = []
            if self.context.rules:
                rule_match, all_matched_indices = evaluate_rules_stage(
                    self.context,
                    merged_packet,
                    frame_packet  # Pass frame for scenario-based rules
                )
            
            # Convert detections to dict for drawing/events (backward compatibility)
            detections = merged_packet.to_dict()
            
            # Draw and publish frame (if shared_store available)
            processed_frame = self._draw_and_publish_frame(
                frame_packet,
                merged_packet,
                detections,
                all_matched_indices
            )
            
            # Handle rule events (rules can use traditional handlers or scenario implementations)
            if rule_match:
                self._handle_event(rule_match, processed_frame, frame_packet, detections, fps)
            
            # Heartbeat
            try:
                from bson import ObjectId
                self.tasks_collection.update_one(
                    {"_id": ObjectId(self.context.task_id)},
                    {"$set": {"updated_at": utc_now()}}
                )
            except Exception:
                pass
    
    def run_patrol(self) -> None:
        """Run pipeline in patrol mode (sleep interval, then process window, repeat)."""
        interval_seconds = max(0, int(self.context.interval_minutes) * 60)
        window_seconds = max(1, int(self.context.check_duration_seconds))
        
        # Check if any rules are counting rules (box_count or class_count)
        # For counting rules, use camera FPS instead of configured FPS
        use_camera_fps = False
        if self.context.rules:
            for rule in self.context.rules:
                rule_type = str(rule.get("type", "")).strip().lower()
                if rule_type in ['box_count', 'class_count']:
                    use_camera_fps = True
                    break
        
        # Default to configured FPS
        fps = self.context.fps
        
        print(f"[worker {self.context.task_id}] 💤 Patrol mode | sleep={interval_seconds}s window={window_seconds}s fps={fps}")
        
        while True:
            # Sleep with heartbeat and stop checks
            if self._sleep_with_heartbeat(interval_seconds):
                return
            
            # Detection window
            window_end = time.time() + window_seconds
            min_interval = 1.0 / max(1, fps)
            next_tick = time.time()
            print(f"[worker {self.context.task_id}] 🔎 Patrol window started ({window_seconds}s)")
            
            # Reset per-window state
            self.context.rule_state = {}
            self.context.frame_index = 0
            last_status = time.time()
            processed_in_window = 0
            skipped_in_window = 0
            self.context.last_seen_hub_index = None
            
            while time.time() < window_end:
                # Check stop conditions
                stop_reason = self.context.get_stop_condition(self.tasks_collection)
                if stop_reason:
                    status = "cancelled" if stop_reason == "stop_requested" else "completed"
                    self.context.update_status(self.tasks_collection, status)
                    print(f"[worker {self.context.task_id}] ⏹️ Stopping (reason={stop_reason})")
                    return
                
                # Stage 1: Acquire frame
                frame_packet = acquire_frame_stage(self.source)
                if frame_packet is None:
                    time.sleep(0.05)
                    continue
                
                # For counting rules, use camera FPS if available
                if use_camera_fps and frame_packet.fps is not None and frame_packet.fps > 0:
                    fps = frame_packet.fps
                    min_interval = 1.0 / max(1, fps)
                
                # FPS pacing
                now_timestamp = time.time()
                if now_timestamp < next_tick:
                    time.sleep(max(0.0, next_tick - now_timestamp))
                next_tick = max(next_tick + min_interval, time.time())
                
                # Update context tracking
                hub_index = frame_packet.frame_index
                if self.context.last_seen_hub_index is not None and hub_index > self.context.last_seen_hub_index + 1:
                    skipped_in_window += (hub_index - self.context.last_seen_hub_index - 1)
                self.context.last_seen_hub_index = hub_index
                self.context.frame_index += 1
                processed_in_window += 1
                
                # Status logging
            if (time.time() - last_status) >= 1.0:
                base_fps = frame_packet.fps if frame_packet.fps is not None else 0.0
                hub_index_status = self.context.last_seen_hub_index or 0
                print(f"[worker {self.context.task_id}] ⏱️ sampling(agent patrol) agent_fps={fps} base_fps={base_fps} processed={processed_in_window}/s skipped={skipped_in_window} hub_frame_index={hub_index_status} camera_id={self.context.camera_id}")
                processed_in_window = 0
                skipped_in_window = 0
                last_status = time.time()
                
                # Stage 2: Inference
                detection_packets = inference_stage(self.context, frame_packet, self.models)
                
                # Stage 3: Merge detections
                merged_packet = merge_detections_stage(detection_packets)
                
                # Stage 4: Evaluate rules (uses rules field from database)
                # Rules can use traditional handlers OR scenario implementations internally
                rule_match = None
                all_matched_indices = []
                if self.context.rules:
                    rule_match, all_matched_indices = evaluate_rules_stage(
                        self.context,
                        merged_packet,
                        frame_packet  # Pass frame for scenario-based rules
                    )
                
                # Convert detections to dict for drawing/events
                detections = merged_packet.to_dict()
                
                # Draw and publish frame
                processed_frame = self._draw_and_publish_frame(
                    frame_packet,
                    merged_packet,
                    detections,
                    all_matched_indices
                )
                
                # Handle rule events (rules can use traditional handlers or scenario implementations)
                if rule_match:
                    self._handle_event(rule_match, processed_frame, frame_packet, detections, fps)
                
                # Heartbeat
                try:
                    from bson import ObjectId
                    self.tasks_collection.update_one(
                        {"_id": ObjectId(self.context.task_id)},
                        {"$set": {"updated_at": utc_now()}}
                    )
                except Exception:
                    pass
            
            print(f"[worker {self.context.task_id}] 💤 Patrol window ended; going back to sleep")
    
    def _sleep_with_heartbeat(self, seconds: int) -> bool:
        """Sleep with periodic heartbeat and stop checks. Returns True if should stop."""
        from bson import ObjectId
        
        end_time = time.time() + max(0, seconds)
        while time.time() < end_time:
            # Heartbeat
            try:
                self.tasks_collection.update_one(
                    {"_id": ObjectId(self.context.task_id)},
                    {"$set": {"updated_at": utc_now()}}
                )
            except Exception:
                pass
            
            # Check stop condition
            stop_reason = self.context.get_stop_condition(self.tasks_collection)
            if stop_reason:
                status = "cancelled" if stop_reason == "stop_requested" else "completed"
                self.context.update_status(self.tasks_collection, status)
                print(f"[worker {self.context.task_id}] ⏹️ Stopping (reason={stop_reason})")
                return True
            
            time.sleep(1)
        return False
    
    def _draw_and_publish_frame(
        self,
        frame_packet: FramePacket,
        merged_packet: DetectionPacket,
        detections: Dict[str, Any],
        all_matched_indices: List[int]
    ) -> Optional[Any]:
        """Draw bounding boxes and publish processed frame to shared_store."""
        if self.shared_store is None or not self.context.rules:
            return None
        
        try:
            frame = frame_packet.frame.copy()
            
            # For restricted zone scenarios: show ALL person detections, not just those in zone
            # For class_count/box_count with line: show ALL detections of target class
            # For other scenarios: show only matched detections
            zone_violated = False
            target_class = None
            line_crossed_indices = []  # Indices of objects that crossed the line
            line_zone = None  # Line zone data for visualization
            track_info = []  # Track information (center points, track IDs) for visualization
            counts = None  # Counts for box/class counting scenarios
            active_tracks_count = 0  # Number of active tracks
            
            # Check scenario types
            is_restricted_zone = False
            is_line_counting = False
            is_fire_detection = False
            fire_detected = False  # Whether fire is currently detected
            fire_classes = []  # Target classes for fire detection
            
            if hasattr(self.context, '_scenario_instances'):
                for rule_idx, scenario_instance in self.context._scenario_instances.items():
                    scenario_type = getattr(scenario_instance, 'scenario_id', '')
                    
                    # Check if this is a restricted zone scenario
                    if scenario_type == 'restricted_zone':
                        is_restricted_zone = True
                        if hasattr(scenario_instance, 'config_obj') and hasattr(scenario_instance.config_obj, 'target_class'):
                            target_class = scenario_instance.config_obj.target_class
                        # Check if zone is violated
                        if hasattr(scenario_instance, '_state'):
                            state = scenario_instance._state
                            if state.get('objects_in_zone', False):
                                zone_violated = True
                    
                    # Check if this is a fire detection scenario
                    elif scenario_type == 'fire_detection':
                        is_fire_detection = True
                        if hasattr(scenario_instance, 'config_obj') and hasattr(scenario_instance.config_obj, 'target_classes'):
                            fire_classes = scenario_instance.config_obj.target_classes
                        # Check if fire is currently detected (from scenario state)
                        if hasattr(scenario_instance, '_state'):
                            state = scenario_instance._state
                            # Fire is detected if we have consecutive frames with fire above threshold
                            if state.get('fire_detected', False) or state.get('consecutive_fire_frames', 0) >= 1:
                                fire_detected = True
                    
                    # Check if this is a line-based counting scenario (class_count or box_count)
                    elif scenario_type in ['class_count', 'box_count']:
                        is_line_counting = True
                        
                        if hasattr(scenario_instance, 'config_obj') and hasattr(scenario_instance.config_obj, 'target_class'):
                            target_class = scenario_instance.config_obj.target_class
                        
                        # Get line zone data for visualization
                        if hasattr(scenario_instance, 'config_obj') and hasattr(scenario_instance.config_obj, 'zone_type'):
                            if scenario_instance.config_obj.zone_type == 'line' and scenario_instance.config_obj.zone_coordinates:
                                line_zone = {
                                    "type": "line",
                                    "coordinates": scenario_instance.config_obj.zone_coordinates,
                                    "direction": scenario_instance.config_obj.zone_direction
                                }
                        
                        # Get objects that touched the line (for visualization)
                        if hasattr(scenario_instance, '_state'):
                            state = scenario_instance._state
                            
                            # Get track info with center points and touch status for visualization
                            scenario_track_info = state.get('track_info', [])
                            
                            if scenario_track_info:
                                track_info = scenario_track_info.copy()  # Make a copy to avoid reference issues
                            else:
                                track_info = []  # Ensure track_info is always a list
                            
                            # Get counts from line counter if available
                            if hasattr(scenario_instance, 'line_counter') and scenario_instance.line_counter:
                                counts = scenario_instance.line_counter.get_counts()
                            
                            # Get active tracks count
                            if hasattr(scenario_instance, 'tracker') and scenario_instance.tracker:
                                all_active_tracks = scenario_instance.tracker.get_all_active_tracks()
                                active_tracks_count = len(all_active_tracks)
                            
                            # Match tracks that are currently touching the line to detection indices
                            # Use track_info which already has touching_line flag (more accurate than matching touched_track_ids)
                            if track_info and hasattr(scenario_instance, 'tracker') and scenario_instance.tracker:
                                # Get active tracks from tracker for matching
                                active_tracks = scenario_instance.tracker.tracks
                                
                                # Create a map of track_id to track_info for quick lookup
                                track_info_map = {t.get('track_id'): t for t in track_info if t.get('track_id') is not None}
                                
                                # Match tracks that are touching the line to detection indices by IoU
                                for track in active_tracks:
                                    track_info_item = track_info_map.get(track.track_id)
                                    # Only match tracks that are currently touching the line
                                    if track_info_item and track_info_item.get('touching_line', False):
                                        # Find matching detection index
                                        track_bbox = track.bbox
                                        for idx, det_box in enumerate(merged_packet.boxes):
                                            if idx >= len(merged_packet.classes):
                                                continue
                                            # Check if class matches
                                            det_class = merged_packet.classes[idx]
                                            if isinstance(det_class, str) and det_class.lower() == target_class:
                                                # Calculate IoU
                                                iou = self._calculate_iou(track_bbox, det_box)
                                                if iou >= 0.3:  # Match threshold
                                                    if idx not in line_crossed_indices:
                                                        line_crossed_indices.append(idx)
                                                    break
            
            # Draw annotations based on scenario type
            # For line-based counting (box_count/class_count), frontend handles drawing
            # Backend only draws for pose keypoints or regular bounding boxes
            if detections.get("keypoints"):
                processed_frame = draw_pose_keypoints(frame, detections, self.context.rules)
            elif is_line_counting:
                # For line counting, frontend draws boxes/center points based on track_info
                # Don't draw boxes here to avoid duplicates
                processed_frame = frame.copy()
            else:
                processed_frame = draw_bounding_boxes(frame, detections, self.context.rules)
            
            # Convert to bytes
            frame_bytes = processed_frame.tobytes()
            height, width = processed_frame.shape[0], processed_frame.shape[1]
            
            # For restricted zone: indices into filtered detections that are inside the zone (for per-box red coloring)
            in_zone_indices: List[int] = []

            # Filter detections based on scenario type
            if is_restricted_zone and target_class:
                # Show ALL detections of target class (e.g., all persons)
                f_boxes, f_classes, f_scores = self._filter_detections_by_class(
                    target_class, merged_packet.boxes, merged_packet.classes, merged_packet.scores
                )
                # Map merged_packet indices (in zone) to filtered indices for overlay coloring
                orig_in_zone = set(all_matched_indices)
                filtered_idx = 0
                for orig_idx in range(len(merged_packet.boxes)):
                    if orig_idx < len(merged_packet.classes):
                        det_class = merged_packet.classes[orig_idx]
                        if isinstance(det_class, str) and det_class.lower() == target_class:
                            if orig_idx in orig_in_zone:
                                in_zone_indices.append(filtered_idx)
                            filtered_idx += 1
            elif is_fire_detection and fire_classes:
                # Show ALL fire-related detections (fire, flame, smoke)
                # This ensures fire bounding boxes are always visible
                f_boxes, f_classes, f_scores = self._filter_detections_by_fire_classes(
                    fire_classes, merged_packet.boxes, merged_packet.classes, merged_packet.scores
                )
            elif is_line_counting and target_class:
                # Show ALL detections of target class for line counting
                f_boxes, f_classes, f_scores = self._filter_detections_by_class(
                    target_class, merged_packet.boxes, merged_packet.classes, merged_packet.scores
                )
                
                # Map line_crossed_indices from merged_packet indices to filtered indices
                if line_crossed_indices:
                    # Create mapping: original index -> filtered index
                    filtered_crossed_indices = []
                    original_to_filtered = {}
                    filtered_idx = 0
                    for orig_idx in range(len(merged_packet.boxes)):
                        if orig_idx < len(merged_packet.classes):
                            det_class = merged_packet.classes[orig_idx]
                            if isinstance(det_class, str) and det_class.lower() == target_class:
                                if orig_idx in line_crossed_indices:
                                    filtered_crossed_indices.append(filtered_idx)
                                original_to_filtered[orig_idx] = filtered_idx
                                filtered_idx += 1
                    line_crossed_indices = filtered_crossed_indices
            elif all_matched_indices:
                # Other scenarios: show only matched detections
                unique_indices = sorted(set(all_matched_indices))
                f_boxes, f_classes, f_scores = self._filter_detections_by_indices(
                    unique_indices, merged_packet.boxes, merged_packet.classes, merged_packet.scores
                )
            else:
                f_boxes, f_classes, f_scores = [], [], []
            
            # Publish to shared_store
            self.shared_store[self.context.agent_id] = {
                "shape": (height, width, 3),
                "dtype": "uint8",
                "frame_index": frame_packet.frame_index,
                "ts_monotonic": time.time(),
                "camera_fps": frame_packet.fps,
                "actual_fps": frame_packet.fps if frame_packet.fps is not None else self.context.fps,
                "bytes": frame_bytes,
                "agent_id": self.context.agent_id,
                "task_name": self.context.agent_name,
                "detections": {
                    "boxes": f_boxes,
                    "classes": f_classes,
                    "scores": f_scores,
                },
                "rules": self.context.rules,
                "camera_id": self.context.camera_id,
                "zone_violated": zone_violated,  # Zone violation status
                "line_zone": line_zone,  # Line zone data for visualization
                "line_crossed": len(line_crossed_indices) > 0,  # Whether any object crossed the line
                "line_crossed_indices": line_crossed_indices,  # Indices of filtered detections that crossed the line
                "track_info": track_info,  # Track information (center points, track IDs) for visualization
                "fire_detected": fire_detected,  # Fire detection status (for red bounding boxes)
                "in_zone_indices": in_zone_indices,  # Restricted zone: indices of filtered detections inside zone (red box only these)
            }
            
            return processed_frame
        except Exception as exc:  # noqa: BLE001
            print(f"[worker {self.context.task_id}] ⚠️  Error processing frame for stream: {exc}")
            return None
    
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
    
    def _filter_detections_by_indices(
        self,
        indices: List[int],
        boxes: List[List[float]],
        classes: List[str],
        scores: List[float],
    ) -> tuple[List[List[float]], List[str], List[float]]:
        """Filter detections to only those at specified indices with confidence >= 0.7."""
        if not indices:
            return [], [], []
        f_boxes: List[List[float]] = []
        f_classes: List[str] = []
        f_scores: List[float] = []
        confidence_threshold = 0.7  # Minimum confidence level
        
        for idx in indices:
            if 0 <= idx < len(boxes) and idx < len(classes) and idx < len(scores):
                # Apply confidence threshold
                if scores[idx] >= confidence_threshold:
                    f_boxes.append(boxes[idx])
                    f_classes.append(classes[idx])
                    f_scores.append(scores[idx])
        return f_boxes, f_classes, f_scores
    
    def _filter_detections_by_class(
        self,
        target_class: str,
        boxes: List[List[float]],
        classes: List[str],
        scores: List[float],
    ) -> tuple[List[List[float]], List[str], List[float]]:
        """Filter detections to only those matching target class (e.g., 'person') with confidence >= 0.7."""
        f_boxes: List[List[float]] = []
        f_classes: List[str] = []
        f_scores: List[float] = []
        target_lower = target_class.lower()
        confidence_threshold = 0.7  # Minimum confidence level
        
        # Debug: Log all detected classes
        if len(classes) > 0:
            unique_classes = set(cls.lower() if isinstance(cls, str) else str(cls) for cls in classes)
            print(f"[worker {self.context.task_id}] 🔍 YOLO detected classes: {unique_classes} (total: {len(classes)} objects)")
        
        filtered_count = 0
        for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if isinstance(cls, str) and cls.lower() == target_lower:
                # Apply confidence threshold
                if score >= confidence_threshold:
                    f_boxes.append(box)
                    f_classes.append(cls)
                    f_scores.append(score)
                else:
                    filtered_count += 1
        
        # Debug: Log filtering results
        if len(f_boxes) > 0:
            print(f"[worker {self.context.task_id}] ✅ Found {len(f_boxes)} '{target_class}' detections (confidence >= {confidence_threshold})")
        if filtered_count > 0:
            print(f"[worker {self.context.task_id}] 🔽 Filtered out {filtered_count} low-confidence '{target_class}' detections (< {confidence_threshold})")
        
        return f_boxes, f_classes, f_scores
    
    def _filter_detections_by_fire_classes(
        self,
        fire_classes: List[str],
        boxes: List[List[float]],
        classes: List[str],
        scores: List[float],
    ) -> tuple[List[List[float]], List[str], List[float]]:
        """
        Filter detections to only fire-related classes (fire, flame, smoke, etc.).
        
        Uses a lower confidence threshold (0.5) for fire detection since it's safety-critical.
        """
        f_boxes: List[List[float]] = []
        f_classes: List[str] = []
        f_scores: List[float] = []
        fire_classes_lower = [c.lower() for c in fire_classes]
        confidence_threshold = 0.5  # Lower threshold for fire detection (safety-critical)
        
        for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if isinstance(cls, str):
                cls_lower = cls.lower()
                # Check if class matches any fire-related class
                is_fire_class = any(
                    target in cls_lower or cls_lower in target
                    for target in fire_classes_lower
                )
                
                if is_fire_class and score >= confidence_threshold:
                    f_boxes.append(box)
                    f_classes.append(cls)
                    f_scores.append(score)
        
        # Debug: Log fire detections
        if len(f_boxes) > 0:
            print(f"[worker {self.context.task_id}] 🔥 Found {len(f_boxes)} fire-related detections (confidence >= {confidence_threshold})")
        
        return f_boxes, f_classes, f_scores
    
    def _handle_event(
        self,
        rule_match: RuleMatch,
        processed_frame: Optional[Any],
        frame_packet: FramePacket,
        detections: Dict[str, Any],
        fps: int
    ) -> None:
        """Handle rule match event (drawing, notifications). DEPRECATED: Use scenarios instead."""
        event_label = rule_match.label
        video_ms = (self.context.frame_index / float(max(1, self.context.fps))) * 1000.0
        video_ts = format_video_time_ms(video_ms)
        
        print(f"[worker {self.context.task_id}] 🔔 {event_label} | agent='{self.context.agent_name}' | video_time={video_ts}")
        
        # Include report in detections if present (contains VLM description, weapon_type, etc.)
        if rule_match.report and detections is not None:
            detections = detections.copy()
            detections["rule_report"] = rule_match.report
        
        # Handle event through session manager
        if processed_frame is not None:
            try:
                session_manager = get_event_session_manager()
                session_manager.handle_event_frame(
                    agent_id=self.context.agent_id,
                    rule_index=rule_match.rule_index,
                    event_label=event_label,
                    frame=processed_frame,
                    camera_id=self.context.camera_id,
                    agent_name=self.context.agent_name,
                    detections=detections,
                    video_timestamp=video_ts,
                    fps=fps
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[worker {self.context.task_id}] ⚠️  Error handling event frame: {exc}")
        elif self.context.rules:
            # Create processed frame if not already created
            try:
                frame = frame_packet.frame.copy()
                if detections.get("keypoints"):
                    processed_frame = draw_pose_keypoints(frame, detections, self.context.rules)
                else:
                    processed_frame = draw_bounding_boxes(frame, detections, self.context.rules)
                
                if rule_match.report and detections is not None:
                    detections = detections.copy()
                    detections["rule_report"] = rule_match.report
                
                session_manager = get_event_session_manager()
                session_manager.handle_event_frame(
                    agent_id=self.context.agent_id,
                    rule_index=rule_match.rule_index,
                    event_label=event_label,
                    frame=processed_frame,
                    camera_id=self.context.camera_id,
                    agent_name=self.context.agent_name,
                    detections=detections,
                    video_timestamp=video_ts,
                    fps=fps
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[worker {self.context.task_id}] ⚠️  Error creating/handling event frame: {exc}")
