"""
Pipeline Runner and Stages
--------------------------

Orchestrates the processing pipeline and defines the individual stages.
Manages the main processing loop, FPS pacing, and stage sequencing.
"""

import dataclasses
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.processing.processing_output.data_models import DetectionPacket
from app.processing.models.data_models import Model
from app.processing.pipeline.context import PipelineContext
from app.processing.pipeline.data_models import RuleMatch
from app.processing.processing_output import convert_from_yolo_result
from app.processing.processing_output.merger import DetectionMerger
from app.processing.vision_tasks.data_models import ScenarioFrameContext
from app.processing.vision_tasks.task_lookup import get_scenario_class
from app.processing.data_input.data_models import FramePacket
from app.processing.data_input.source_factory import Source
from app.processing.drawing.frame_processor import (
    draw_bounding_boxes,
    draw_pose_keypoints,
    draw_box_count_annotations,
)
from app.utils.db import get_collection
from app.utils.datetime_utils import utc_now, now
from app.utils.event_session_manager import get_event_session_manager

# Import scenarios module to trigger registration decorators
from app.processing.vision_tasks import scenario_registry  # noqa: F401

# ============================================================================
# PIPELINE STAGES
# ============================================================================

def acquire_frame_stage(source: Source) -> Optional[FramePacket]:
    """Stage 1: Acquire frame from source."""
    return source.read_frame()

def _any_scenario_requires_yolo(context: PipelineContext) -> bool:
    """Check if any scenario in the rules requires YOLO detections."""
    if not context.rules:
        return False

    if hasattr(context, '_scenario_instances'):
        for scenario_instance in context._scenario_instances.values():
            if scenario_instance.requires_yolo_detections():
                return True

    for rule_idx, rule in enumerate(context.rules):
        rule_type = str(rule.get("type", "")).strip().lower()
        scenario_class = get_scenario_class(rule_type)

        if scenario_class is None:
            continue

        if hasattr(context, '_scenario_instances') and rule_idx in context._scenario_instances:
            if context._scenario_instances[rule_idx].requires_yolo_detections():
                return True
        else:
            try:
                scenario_config = {k: v for k, v in rule.items() if k != "type"}
                temp_instance = scenario_class(scenario_config, context)
                if temp_instance.requires_yolo_detections():
                    return True
            except Exception:
                return True

    return False

def inference_stage(
    context: PipelineContext,
    frame_packet: FramePacket,
    models: List[Model]
) -> List[DetectionPacket]:
    """Stage 2: Run model inference on frame."""
    if not _any_scenario_requires_yolo(context):
        return []

    frame = frame_packet.frame
    detection_packets: List[DetectionPacket] = []

    for model in models:
        try:
            results = model(frame, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker {context.task_id}] ⚠️ YOLO error: {exc}")
            continue
        if results:
            first_result = results[0]
            packet = convert_from_yolo_result(first_result, now())
            detection_packets.append(packet)

    return detection_packets

def merge_detections_stage(detection_packets: List[DetectionPacket]) -> DetectionPacket:
    """Stage 3: Merge detections from multiple models."""
    return DetectionMerger.merge(detection_packets, now())


def _overlay_data_to_dict_list(overlay_data: Any) -> List[Dict[str, Any]]:
    """Normalize scenario overlay output for UI: each item has boxes, labels, colors arrays."""
    if isinstance(overlay_data, list):
        out = []
        for item in overlay_data:
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                item = dataclasses.asdict(item)
            if isinstance(item, dict):
                # UI expects overlay.boxes, overlay.labels, overlay.colors (arrays)
                if "boxes" in item:
                    out.append(item)
                elif "box" in item:
                    out.append({
                        "boxes": [item["box"]],
                        "labels": [item.get("label") or ""],
                        "colors": [item.get("color") or [0, 255, 102]],
                    })
                else:
                    out.append(item)
            else:
                out.append(item)
        return out
    if isinstance(overlay_data, dict):
        # Loom-style: {boxes, labels, colors} -> list of {box, label, color}
        boxes = overlay_data.get("boxes") or []
        labels = overlay_data.get("labels") or []
        colors = overlay_data.get("colors") or []
        n = max(len(boxes), len(labels), len(colors))
        return [
            {
                "box": boxes[i] if i < len(boxes) else None,
                "label": labels[i] if i < len(labels) else None,
                "color": colors[i] if i < len(colors) else None,
            }
            for i in range(n)
        ]
    return []


def evaluate_rules_stage(
    context: PipelineContext,
    merged_packet: DetectionPacket,
    frame_packet: Optional[FramePacket] = None
) -> tuple[Optional[RuleMatch], List[int]]:
    """
    Stage 4: Evaluate rules against detections.

    All rules now use scenario implementations.
    Each rule type must have a corresponding scenario registered.
    """
    if not context.rules:
        return None, []

    detections = merged_packet.to_dict()

    if not hasattr(context, '_scenario_instances'):
        context._scenario_instances = {}

    all_matched_indices: List[int] = []
    event = None

    for rule_idx, rule in enumerate(context.rules):
        rule_type = str(rule.get("type", "")).strip().lower()

        scenario_class = get_scenario_class(rule_type)
        if scenario_class is None:
            print(f"[evaluate_rules_stage] ⚠️  No scenario found for rule type '{rule_type}'. Skipping.")
            continue

        if frame_packet is None:
            print(f"[evaluate_rules_stage] ⚠️  Frame packet required for scenario '{rule_type}'. Skipping.")
            continue

        try:
            if rule_idx not in context._scenario_instances:
                scenario_config = {k: v for k, v in rule.items() if k != "type"}
                scenario_instance = scenario_class(scenario_config, context)
                # Explicitly set scenario_id from rule type (like old code) so pipeline
                # recognizes box_count/class_count for line counting and track_info
                scenario_instance.scenario_id = rule_type
                context._scenario_instances[rule_idx] = scenario_instance

            scenario_instance = context._scenario_instances[rule_idx]

            # Use wall-clock time for scenarios (frame_packet.timestamp is monotonic; fromtimestamp can produce
            # 1970-era datetimes that cause OSError [Errno 22] on Windows when .timestamp() is used in reporters)
            frame_timestamp = utc_now()

            frame_context = ScenarioFrameContext(
                frame=frame_packet.frame,
                frame_index=context.frame_index,
                timestamp=frame_timestamp,
                detections=merged_packet,
                rule_matches=[],
                pipeline_context=context
            )

            scenario_events = scenario_instance.process(frame_context)

            if scenario_events:
                scenario_event = scenario_events[0]

                report = None
                if scenario_event.metadata:
                    report = scenario_event.metadata.get("report")
                    if report is None:
                        report = {k: v for k, v in scenario_event.metadata.items() if k != "report"}

                rule_result = {
                    "label": scenario_event.label,
                    "matched_detection_indices": scenario_event.detection_indices,
                    "report": report
                }

                if rule_result:
                    idx_list = rule_result.get("matched_detection_indices")
                    if isinstance(idx_list, list):
                        all_matched_indices.extend(idx_list)
                    if not event and rule_result.get("label"):
                        event = rule_result
                        event.setdefault("rule_index", rule_idx)
        except Exception as exc:
            print(f"[evaluate_rules_stage] ⚠️  Error processing scenario '{rule_type}': {exc}")
            continue

    rule_match = None
    if event and event.get("label"):
        rule_match = RuleMatch(
            label=str(event["label"]).strip(),
            rule_index=event.get("rule_index", 0),
            matched_detection_indices=event.get("matched_detection_indices", []),
            report=event.get("report")
        )

    return rule_match, all_matched_indices

# ============================================================================
# UTILITIES
# ============================================================================

def format_video_time_ms(milliseconds: float) -> str:
    """Convert milliseconds to H:MM:SS.mmm string."""
    if milliseconds < 0:
        milliseconds = 0
    total_seconds, milliseconds_remainder = divmod(int(milliseconds), 1000)
    hours, remaining_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{milliseconds_remainder:03d}"

# ============================================================================
# PIPELINE RUNNER
# ============================================================================

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
                all_matched_indices,
                rule_match,
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

                if (time.time() - last_status) >= 1.0:
                    base_fps = frame_packet.fps if frame_packet.fps is not None else 0.0
                    hub_index_status = self.context.last_seen_hub_index or 0
                    print(f"[worker {self.context.task_id}] ⏱️ sampling(agent patrol) agent_fps={fps} base_fps={base_fps} processed={processed_in_window}/s skipped={skipped_in_window} hub_frame_index={hub_index_status} camera_id={self.context.camera_id}")
                    processed_in_window = 0
                    skipped_in_window = 0
                    last_status = time.time()

                detection_packets = inference_stage(self.context, frame_packet, self.models)
                merged_packet = merge_detections_stage(detection_packets)

                rule_match = None
                all_matched_indices = []
                if self.context.rules:
                    rule_match, all_matched_indices = evaluate_rules_stage(
                        self.context,
                        merged_packet,
                        frame_packet
                    )

                detections = merged_packet.to_dict()
                processed_frame = self._draw_and_publish_frame(
                    frame_packet,
                    merged_packet,
                    detections,
                    all_matched_indices,
                    rule_match,
                )

                if rule_match:
                    self._handle_event(rule_match, processed_frame, frame_packet, detections, fps)

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
        all_matched_indices: List[int],
        rule_match: Optional[RuleMatch] = None,
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
            restricted_zone_scenario = None  # Used to read in_zone_indices from state for per-box red coloring
            is_line_counting = False
            is_fire_detection = False
            fire_detected = False  # Whether fire is currently detected
            fire_classes = []  # Target classes for fire detection
            is_weapon_detection = False  # Show person + keypoints for pose/skeleton overlay
            is_sleep_detection = False  # Show person + keypoints for pose/skeleton overlay (sleep detection)
            sleep_detection_scenario = None  # For per-box red when sleep confirmed (same box, change color)
            is_wall_climb_detection = False  # Orange = climbing, red = fully above (stays red)
            wall_climb_scenario = None
            face_recognitions: List[Dict[str, Any]] = []  # Face detection: [{ "box": [...], "name": "..." }]
            
            if hasattr(self.context, '_scenario_instances'):
                for rule_idx, scenario_instance in self.context._scenario_instances.items():
                    scenario_type = getattr(scenario_instance, 'scenario_id', '')
                    
                    # Check if this is a restricted zone scenario
                    if scenario_type == 'restricted_zone':
                        is_restricted_zone = True
                        restricted_zone_scenario = scenario_instance
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
                    
                    # Check if this is weapon_detection (show person + keypoints for pose overlay)
                    elif scenario_type == 'weapon_detection':
                        is_weapon_detection = True
                    # Check if this is sleep_detection (show person + keypoints; red box when VLM confirms sleep)
                    elif scenario_type == 'sleep_detection':
                        is_sleep_detection = True
                        sleep_detection_scenario = scenario_instance
                    # Wall climb: orange = climbing, red = fully above (stays red)
                    elif scenario_type == 'wall_climb_detection':
                        is_wall_climb_detection = True
                        wall_climb_scenario = scenario_instance
                        if hasattr(scenario_instance, 'config_obj') and hasattr(scenario_instance.config_obj, 'target_class'):
                            target_class = scenario_instance.config_obj.target_class
                        if hasattr(scenario_instance, '_state'):
                            state = scenario_instance._state
                            if state.get('red_indices') or state.get('climbing_indices'):
                                zone_violated = True
                    # Face detection: recognized faces (box + name) for overlay
                    elif scenario_type == 'face_detection':
                        if hasattr(scenario_instance, '_state'):
                            face_recognitions = list(scenario_instance._state.get("recognized_faces") or [])
                    
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
                # For line counting we draw below using track_info from state
                processed_frame = frame.copy()
            else:
                processed_frame = draw_bounding_boxes(frame, detections, self.context.rules)
            
            # For line-based counting (class_count or box_count): draw track_info + ADD/OUT/TRACKING
            # Use track_info from state (like old code) so colors update properly when box crosses line
            # Same behavior for both box_count and class_count: green/orange/yellow by counted + direction
            if is_line_counting and track_info:
                draw_counts = counts or {}
                processed_frame = draw_box_count_annotations(
                    processed_frame,
                    track_info,
                    None,
                    {
                        "entry_count": draw_counts.get("entry_count", 0),
                        "exit_count": draw_counts.get("exit_count", 0),
                    },
                    (target_class or "object"),
                    active_tracks_count,
                )
            elif rule_match and rule_match.report:
                # Fallback: draw from report when state track_info not used (e.g. single rule)
                rule_index = getattr(rule_match, "rule_index", None)
                if rule_index is not None and rule_index < len(self.context.rules):
                    rule = self.context.rules[rule_index]
                    report = rule_match.report
                    track_info_from_report = report.get("track_info")
                    if track_info_from_report is not None and not (is_line_counting and track_info):
                        # Use rule/report target class only (no hardcoded box/person)
                        draw_class = (
                            rule.get("target_class")
                            or rule.get("class")
                            or report.get("target_class")
                            or "object"
                        )
                        if isinstance(draw_class, str):
                            draw_class = draw_class.strip() or "object"
                        else:
                            draw_class = "object"
                        processed_frame = draw_box_count_annotations(
                            processed_frame,
                            track_info_from_report,
                            None,
                            {
                                "entry_count": report.get("entry_count", 0),
                                "exit_count": report.get("exit_count", 0),
                            },
                            draw_class,
                            report.get("active_tracks", len(track_info_from_report)),
                        )
            
            # Convert to bytes
            frame_bytes = processed_frame.tobytes()
            height, width = processed_frame.shape[0], processed_frame.shape[1]
            
            # For restricted zone: indices into filtered detections that are inside the zone (for per-box red coloring)
            in_zone_indices: List[int] = []
            # Wall climb: red = fully above (stays red), orange = climbing
            wall_climb_red_indices: List[int] = []
            wall_climb_orange_indices: List[int] = []

            # Filter detections based on scenario type (include keypoints for fall_detection/pose)
            keypoints_src = getattr(merged_packet, "keypoints", None) or []
            if is_restricted_zone and target_class:
                # Show ALL detections of target class (e.g., all persons)
                f_boxes, f_classes, f_scores, f_keypoints = self._filter_detections_by_class(
                    target_class, merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    keypoints_src,
                )
                # Use scenario state in_zone_indices so per-box red is correct even during alert cooldown
                # (when no event is emitted, all_matched_indices is empty but state still has who is in zone)
                state_in_zone = []
                if restricted_zone_scenario and hasattr(restricted_zone_scenario, '_state'):
                    state_in_zone = restricted_zone_scenario._state.get("in_zone_indices") or []
                orig_in_zone = set(state_in_zone) if state_in_zone else set(all_matched_indices)
                filtered_idx = 0
                for orig_idx in range(len(merged_packet.boxes)):
                    if orig_idx < len(merged_packet.classes):
                        det_class = merged_packet.classes[orig_idx]
                        if isinstance(det_class, str) and det_class.lower() == target_class:
                            if orig_idx in orig_in_zone:
                                in_zone_indices.append(filtered_idx)
                            filtered_idx += 1
            elif is_wall_climb_detection and target_class:
                # Show ALL detections of target class (e.g., person); color by climbing / fully above
                f_boxes, f_classes, f_scores, f_keypoints = self._filter_detections_by_class(
                    target_class, merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    keypoints_src,
                )
                state_red = []
                state_climbing = []
                if wall_climb_scenario and hasattr(wall_climb_scenario, '_state'):
                    state_red = wall_climb_scenario._state.get("red_indices") or []
                    state_climbing = wall_climb_scenario._state.get("climbing_indices") or []
                orig_red = set(state_red)
                orig_climbing = set(state_climbing)
                filtered_idx = 0
                for orig_idx in range(len(merged_packet.boxes)):
                    if orig_idx < len(merged_packet.classes):
                        det_class = merged_packet.classes[orig_idx]
                        if isinstance(det_class, str) and det_class.lower() == target_class:
                            if orig_idx in orig_red:
                                wall_climb_red_indices.append(filtered_idx)
                            elif orig_idx in orig_climbing:
                                wall_climb_orange_indices.append(filtered_idx)
                            filtered_idx += 1
            elif is_fire_detection and fire_classes:
                # Show ALL fire-related detections (fire, flame, smoke)
                f_boxes, f_classes, f_scores, f_keypoints = self._filter_detections_by_fire_classes(
                    fire_classes, merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    keypoints_src,
                )
            elif is_weapon_detection or is_sleep_detection:
                # Show ALL person detections with keypoints so UI can draw pose/skeleton
                f_boxes, f_classes, f_scores, f_keypoints = self._filter_detections_by_class(
                    "person", merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    keypoints_src,
                )
            elif is_line_counting and target_class:
                # Show ALL detections of target class for line counting
                f_boxes, f_classes, f_scores, f_keypoints = self._filter_detections_by_class(
                    target_class, merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    keypoints_src,
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
                # Other scenarios (e.g. fall_detection): show only matched detections; include keypoints for pose
                unique_indices = sorted(set(all_matched_indices))
                f_boxes, f_classes, f_scores, f_keypoints = self._filter_detections_by_indices(
                    unique_indices, merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    keypoints_src,
                )
            else:
                f_boxes, f_classes, f_scores, f_keypoints = [], [], [], []
            
            # Sleep detection: which person boxes are VLM-confirmed sleeping (same box, UI draws red)
            sleep_confirmed_indices: List[int] = []
            if is_sleep_detection and sleep_detection_scenario and hasattr(sleep_detection_scenario, 'state'):
                emitted = getattr(sleep_detection_scenario.state, 'emitted_event_boxes', {})
                for i, box in enumerate(f_boxes):
                    if not box or len(box) < 4:
                        continue
                    for eb in emitted.values():
                        if self._calculate_iou(box, eb) >= 0.3:
                            sleep_confirmed_indices.append(i)
                            break
            
            # Collect scenario overlays (e.g., loom ROI boxes with state labels)
            # Normalize to list of JSON-serializable dicts for WebSocket payload
            scenario_overlays = []
            if hasattr(self.context, '_scenario_instances'):
                frame_timestamp = utc_now()
                frame_context = ScenarioFrameContext(
                    frame=frame_packet.frame,
                    frame_index=self.context.frame_index,
                    timestamp=frame_timestamp,
                    detections=merged_packet,
                    rule_matches=[],
                    pipeline_context=self.context
                )
                for scenario_instance in self.context._scenario_instances.values():
                    scenario_type = getattr(scenario_instance, "scenario_id", "")
                    overlay_data = scenario_instance.get_overlay_data(frame_context)
                    if overlay_data:
                        scenario_overlays.extend(
                            _overlay_data_to_dict_list(overlay_data)
                        )
                        # Face detection: use same (smoothed) overlay for face_recognitions so UI gets boxes consistently
                        if scenario_type == "face_detection":
                            face_recognitions = []
                            for item in overlay_data:
                                if dataclasses.is_dataclass(item) and not isinstance(item, type):
                                    box = getattr(item, "box", None)
                                    name = getattr(item, "label", "Unknown") or "Unknown"
                                    if box and len(box) >= 4:
                                        face_recognitions.append({"box": list(box), "name": name})
                                elif isinstance(item, dict) and item.get("box") and len(item.get("box", [])) >= 4:
                                    face_recognitions.append({
                                        "box": item["box"],
                                        "name": item.get("label", item.get("name", "Unknown")) or "Unknown",
                                    })
            
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
                    "keypoints": f_keypoints,  # For fall_detection/pose UI (skeleton overlay)
                },
                "scenario_overlays": scenario_overlays,  # Add scenario overlays
                "rules": self.context.rules,
                "camera_id": self.context.camera_id,
                "zone_violated": zone_violated,  # Zone violation status
                "line_zone": line_zone,  # Line zone data for visualization
                "line_crossed": len(line_crossed_indices) > 0,  # Whether any object crossed the line
                "line_crossed_indices": line_crossed_indices,  # Indices of filtered detections that crossed the line
                "track_info": track_info,  # Track information (center points, track IDs) for visualization
                "fire_detected": fire_detected,  # Fire detection status (for red bounding boxes)
                "in_zone_indices": in_zone_indices,  # Restricted zone: indices of filtered detections inside zone (red box only these)
                "sleep_confirmed_indices": sleep_confirmed_indices,  # Sleep: same person box, red when VLM confirmed
                "wall_climb_red_indices": wall_climb_red_indices,  # Wall climb: fully above (stays red)
                "wall_climb_orange_indices": wall_climb_orange_indices,  # Wall climb: climbing (orange)
                "face_recognitions": face_recognitions,  # Face detection: [{ "box": [x1,y1,x2,y2], "name": "..." }]
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
        keypoints: Optional[List[List[List[float]]]] = None,
    ) -> tuple[List[List[float]], List[str], List[float], List[List[List[float]]]]:
        """Filter detections to only those at specified indices with confidence >= 0.7.
        When keypoints is provided (e.g. for fall_detection/pose), returns aligned f_keypoints."""
        if not indices:
            return [], [], [], []
        f_boxes: List[List[float]] = []
        f_classes: List[str] = []
        f_scores: List[float] = []
        f_keypoints: List[List[List[float]]] = []
        confidence_threshold = 0.7  # Minimum confidence level
        kp_list = keypoints if keypoints else []

        for idx in indices:
            if 0 <= idx < len(boxes) and idx < len(classes) and idx < len(scores):
                # Apply confidence threshold
                if scores[idx] >= confidence_threshold:
                    f_boxes.append(boxes[idx])
                    f_classes.append(classes[idx])
                    f_scores.append(scores[idx])
                    if idx < len(kp_list):
                        f_keypoints.append(kp_list[idx])
        return f_boxes, f_classes, f_scores, f_keypoints
    
    def _filter_detections_by_class(
        self,
        target_class: str,
        boxes: List[List[float]],
        classes: List[str],
        scores: List[float],
        keypoints: Optional[List[List[List[float]]]] = None,
    ) -> tuple[List[List[float]], List[str], List[float], List[List[List[float]]]]:
        """Filter detections to only those matching target class (e.g., 'person') with confidence >= 0.7.
        When keypoints is provided (e.g. for fall_detection/pose), returns aligned f_keypoints."""
        f_boxes: List[List[float]] = []
        f_classes: List[str] = []
        f_scores: List[float] = []
        f_keypoints: List[List[List[float]]] = []
        target_lower = target_class.lower()
        confidence_threshold = 0.7  # Minimum confidence level
        kp_list = keypoints if keypoints else []

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
                    if idx < len(kp_list):
                        f_keypoints.append(kp_list[idx])
                else:
                    filtered_count += 1
        
        # Debug: Log filtering results
        if len(f_boxes) > 0:
            print(f"[worker {self.context.task_id}] ✅ Found {len(f_boxes)} '{target_class}' detections (confidence >= {confidence_threshold})")
        if filtered_count > 0:
            print(f"[worker {self.context.task_id}] 🔽 Filtered out {filtered_count} low-confidence '{target_class}' detections (< {confidence_threshold})")
        
        return f_boxes, f_classes, f_scores, f_keypoints
    
    def _filter_detections_by_fire_classes(
        self,
        fire_classes: List[str],
        boxes: List[List[float]],
        classes: List[str],
        scores: List[float],
        keypoints: Optional[List[List[List[float]]]] = None,
    ) -> tuple[List[List[float]], List[str], List[float], List[List[List[float]]]]:
        """
        Filter detections to only fire-related classes (fire, flame, smoke, etc.).
        Uses a lower confidence threshold (0.5) for fire detection since it's safety-critical.
        When keypoints is provided, returns aligned f_keypoints (usually empty for fire).
        """
        f_boxes: List[List[float]] = []
        f_classes: List[str] = []
        f_scores: List[float] = []
        f_keypoints: List[List[List[float]]] = []
        fire_classes_lower = [c.lower() for c in fire_classes]
        confidence_threshold = 0.5  # Lower threshold for fire detection (safety-critical)
        kp_list = keypoints if keypoints else []

        for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if isinstance(cls, str):
                cls_lower = cls.lower()
                is_fire_class = any(
                    target in cls_lower or cls_lower in target
                    for target in fire_classes_lower
                )
                if is_fire_class and score >= confidence_threshold:
                    f_boxes.append(box)
                    f_classes.append(cls)
                    f_scores.append(score)
                    if idx < len(kp_list):
                        f_keypoints.append(kp_list[idx])

        if len(f_boxes) > 0:
            print(f"[worker {self.context.task_id}] 🔥 Found {len(f_boxes)} fire-related detections (confidence >= {confidence_threshold})")
        return f_boxes, f_classes, f_scores, f_keypoints
    
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
