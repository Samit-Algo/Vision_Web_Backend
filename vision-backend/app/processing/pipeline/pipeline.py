"""
Pipeline Runner and Stages
--------------------------

Orchestrates the processing pipeline: acquire frame → inference → merge → evaluate rules → draw/publish.
Manages the main loop (continuous or patrol), FPS pacing, and stage sequencing.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import dataclasses
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Processing (stages, context, data models, drawing)
# -----------------------------------------------------------------------------
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
    draw_face_recognitions,
    draw_pose_keypoints,
    draw_box_count_annotations,
    draw_zone_polygon,
    draw_zone_line,
    draw_boxes_in_zone_red,
    draw_loom_machine_polygons,
)
# Trigger scenario registration (decorators run on import)
from app.processing.vision_tasks import scenario_registry  # noqa: F401

# -----------------------------------------------------------------------------
# Utils (DB, datetime, events)
# -----------------------------------------------------------------------------
from app.utils.db import get_collection
from app.utils.datetime_utils import utc_now, now
from app.utils.event_session_manager import get_event_session_manager

# -----------------------------------------------------------------------------
# Pipeline stages (Stage 1–4)
# -----------------------------------------------------------------------------

def acquire_frame_stage(source: Source) -> Optional[FramePacket]:
    """Stage 1: Acquire frame from source."""
    return source.read_frame()

def any_scenario_requires_yolo(context: PipelineContext) -> bool:
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
    if not any_scenario_requires_yolo(context):
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


def overlay_data_to_dict_list(overlay_data: Any) -> List[Dict[str, Any]]:
    """
    Normalize scenario overlay output for UI.
    Converts dataclass/dict from scenarios into a list of dicts with boxes, labels, colors.
    """
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
        # Loom machine state: {polygons, colors} -> list of {polygon, color}
        if "polygons" in overlay_data:
            polygons = overlay_data.get("polygons") or []
            colors = overlay_data.get("colors") or []
            n = max(len(polygons), len(colors))
            return [
                {
                    "polygon": polygons[i] if i < len(polygons) else None,
                    "color": colors[i] if i < len(colors) else None,
                }
                for i in range(n)
            ]
        # Legacy loom-style: {boxes, labels, colors} -> list of {box, label, color}
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

            # Pack this frame's data into the context that scenarios expect (frame, detections, etc.)
            frame_context = ScenarioFrameContext(
                frame=frame_packet.frame,
                frame_index=context.frame_index,
                timestamp=frame_timestamp,
                detections=merged_packet,
                rule_matches=[],
                pipeline_context=context
            )

            # Run the scenario (e.g. fall_detection, intrusion); returns list of ScenarioEvent or []
            scenario_events = scenario_instance.process(frame_context)

            if scenario_events:
                # Use the first event from this scenario
                scenario_event = scenario_events[0]

                # Build report from scenario metadata (for notifications/UI)
                report = None
                if scenario_event.metadata:
                    report = scenario_event.metadata.get("report")
                    if report is None:
                        report = {k: v for k, v in scenario_event.metadata.items() if k != "report"}

                # Convert ScenarioEvent to rule_result dict (label, indices, report, event_type)
                rule_result = {
                    "label": scenario_event.label,
                    "matched_detection_indices": scenario_event.detection_indices,
                    "report": report,
                    "event_type": getattr(scenario_event, "event_type", None) or "",
                }

                if rule_result:
                    # Collect all detection indices that matched any rule (for drawing/highlights)
                    idx_list = rule_result.get("matched_detection_indices")
                    if isinstance(idx_list, list):
                        all_matched_indices.extend(idx_list)
                    # First rule that fires becomes the "winning" event for this frame
                    if not event and rule_result.get("label"):
                        event = rule_result
                        event.setdefault("rule_index", rule_idx)
        except Exception as exc:
            print(f"[evaluate_rules_stage] ⚠️  Error processing scenario '{rule_type}': {exc}")
            continue

    # Wrap the winning event (if any) as a RuleMatch for drawing and handle_event
    rule_match = None
    if event and event.get("label"):
        rule_match = RuleMatch(
            label=str(event["label"]).strip(),
            rule_index=event.get("rule_index", 0),
            matched_detection_indices=event.get("matched_detection_indices", []),
            report=event.get("report"),
            event_type=event.get("event_type") or "",
        )

    return rule_match, all_matched_indices

# -----------------------------------------------------------------------------
# Utilities (time format, FPS helper)
# -----------------------------------------------------------------------------


def format_video_time_ms(milliseconds: float) -> str:
    """Convert milliseconds to H:MM:SS.mmm string (for event timestamps)."""
    if milliseconds < 0:
        milliseconds = 0
    total_seconds, milliseconds_remainder = divmod(int(milliseconds), 1000)
    hours, remaining_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{milliseconds_remainder:03d}"


# -----------------------------------------------------------------------------
# FPS helper
# -----------------------------------------------------------------------------

def should_use_camera_fps(context: PipelineContext) -> bool:
    """True if any rule is box_count or class_count (use camera FPS instead of configured FPS)."""
    if not context.rules:
        return False
    for rule in context.rules:
        rule_type = str(rule.get("type", "")).strip().lower()
        if rule_type in ("box_count", "class_count"):
            return True
    return False

# -----------------------------------------------------------------------------
# Pipeline runner (main loop and helpers)
# -----------------------------------------------------------------------------


class PipelineRunner:
    """
    Orchestrates the processing pipeline.

    Runs the main loop: acquire frame → inference → merge → evaluate rules → draw/publish.
    Supports two modes: continuous (run at target FPS until stop) and patrol (sleep then process window).
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

    # -------------------------------------------------------------------------
    # Private: process one frame (main loop logic)
    # -------------------------------------------------------------------------

    def process_one_frame(
        self,
        frame_packet: FramePacket,
        fps: int,
        min_interval: float,
        next_tick: float,
        last_status: float,
        processed_in_window: int,
        skipped_in_window: int,
    ) -> Tuple[float, float, int, int]:
        """
        Run stages 2–4 on an acquired frame: inference → merge → evaluate → draw/publish → event → heartbeat.
        Updates context (frame_index, last_seen_hub_index) and returns updated pacing counters.
        """
        # FPS pacing: wait until next_tick then advance
        now_ts = time.time()
        if now_ts < next_tick:
            time.sleep(max(0.0, next_tick - now_ts))
        next_tick = max(next_tick + min_interval, time.time())

        # Update context (hub index, frame index, window counters)
        hub_index = frame_packet.frame_index
        if self.context.last_seen_hub_index is not None and hub_index > self.context.last_seen_hub_index + 1:
            skipped_in_window += (hub_index - self.context.last_seen_hub_index - 1)
        self.context.last_seen_hub_index = hub_index
        self.context.frame_index += 1
        processed_in_window += 1

        # Status log once per second (then reset window counters)
        if (time.time() - last_status) >= 1.0:
            base_fps = frame_packet.fps if frame_packet.fps is not None else 0.0
            hub_status = self.context.last_seen_hub_index or 0
            print(
                f"[worker {self.context.task_id}] ⏱️ sampling agent_fps={fps} base_fps={base_fps} "
                f"processed={processed_in_window}/s skipped={skipped_in_window} hub_frame_index={hub_status} camera_id={self.context.camera_id}"
            )
            processed_in_window = 0
            skipped_in_window = 0
            last_status = time.time()

        # Stage 2: Inference
        detection_packets = inference_stage(self.context, frame_packet, self.models)
        # Stage 3: Merge
        merged_packet = merge_detections_stage(detection_packets)

        # Debug: keypoints (pose) on first frame and every 30th
        if self.context.frame_index == 1 or self.context.frame_index % 30 == 0:
            if merged_packet.keypoints:
                print(f"[worker {self.context.task_id}] ✅ Keypoints extracted: {len(merged_packet.keypoints)} person(s) with pose data")
            else:
                print(f"[worker {self.context.task_id}] ⚠️ No keypoints extracted! Check if model is a pose model (e.g., yolov8n-pose.pt)")

        # Stage 4: Evaluate rules (scenario-based)
        rule_match = None
        all_matched_indices: List[int] = []
        if self.context.rules:
            rule_match, all_matched_indices = evaluate_rules_stage(
                self.context, merged_packet, frame_packet
            )

        detections = merged_packet.to_dict()
        processed_frame = self.draw_and_publish_frame(
            frame_packet, merged_packet, detections, all_matched_indices, rule_match
        )
        if rule_match:
            self.handle_event(rule_match, processed_frame, frame_packet, detections, fps)

        # Heartbeat (so DB knows task is still running)
        try:
            from bson import ObjectId
            self.tasks_collection.update_one(
                {"_id": ObjectId(self.context.task_id)},
                {"$set": {"updated_at": utc_now()}},
            )
        except Exception:
            pass

        return (next_tick, last_status, processed_in_window, skipped_in_window)

    # -------------------------------------------------------------------------
    # Public: run pipeline in continuous mode
    # -------------------------------------------------------------------------

    def run_continuous(self) -> None:
        """Run pipeline in continuous mode (process frames indefinitely at target FPS)."""
        use_camera_fps = should_use_camera_fps(self.context)
        fps = self.context.fps
        min_interval = 1.0 / max(1, fps)
        next_tick = time.time()
        self.context.frame_index = 0
        last_status = time.time()
        processed_in_window = 0
        skipped_in_window = 0

        while True:
            # 1. Check if we should stop (user cancelled, task deleted, end_time)
            stop_reason = self.context.get_stop_condition(self.tasks_collection)
            if stop_reason:
                status = "cancelled" if stop_reason == "stop_requested" else "completed"
                self.context.update_status(self.tasks_collection, status)
                print(f"[worker {self.context.task_id}] ⏹️ Stopping (reason={stop_reason})")
                return

            # 2. Stage 1: Acquire frame from source
            frame_packet = acquire_frame_stage(self.source)
            if frame_packet is None:
                if self.context.is_video_file:
                    self.context.update_status(self.tasks_collection, "completed")
                    print(f"[worker {self.context.task_id}] ✅ Video file finished (EOF). Completed.")
                    return
                time.sleep(0.05)
                continue

            # 3. For counting rules, use camera FPS when available
            if use_camera_fps and frame_packet.fps is not None and frame_packet.fps > 0:
                fps = int(frame_packet.fps)
                min_interval = 1.0 / max(1, fps)

            # 4. Run stages 2–4 and draw/publish/event/heartbeat (shared logic)
            try:
                next_tick, last_status, processed_in_window, skipped_in_window = self.process_one_frame(
                    frame_packet, fps, min_interval, next_tick, last_status, processed_in_window, skipped_in_window
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[worker {self.context.task_id}] ⚠️ Error in process_one_frame (continuing): {exc}")
                traceback.print_exc()
                next_tick = max(next_tick + min_interval, time.time())

    # -------------------------------------------------------------------------
    # Public: run pipeline in patrol mode
    # -------------------------------------------------------------------------

    def run_patrol(self) -> None:
        """Run pipeline in patrol mode: sleep N minutes, then process for M seconds, repeat."""
        interval_seconds = max(0, int(self.context.interval_minutes) * 60)
        window_seconds = max(1, int(self.context.check_duration_seconds))
        use_camera_fps = should_use_camera_fps(self.context)
        fps = self.context.fps

        print(f"[worker {self.context.task_id}] 💤 Patrol mode | sleep={interval_seconds}s window={window_seconds}s fps={fps}")

        while True:
            # Sleep interval with heartbeat and stop checks
            if self.sleep_with_heartbeat(interval_seconds):
                return

            # Detection window: process frames for window_seconds
            window_end = time.time() + window_seconds
            min_interval = 1.0 / max(1, fps)
            next_tick = time.time()
            self.context.rule_state = {}
            self.context.frame_index = 0
            last_status = time.time()
            processed_in_window = 0
            skipped_in_window = 0
            self.context.last_seen_hub_index = None
            print(f"[worker {self.context.task_id}] 🔎 Patrol window started ({window_seconds}s)")

            while time.time() < window_end:
                stop_reason = self.context.get_stop_condition(self.tasks_collection)
                if stop_reason:
                    status = "cancelled" if stop_reason == "stop_requested" else "completed"
                    self.context.update_status(self.tasks_collection, status)
                    print(f"[worker {self.context.task_id}] ⏹️ Stopping (reason={stop_reason})")
                    return

                frame_packet = acquire_frame_stage(self.source)
                if frame_packet is None:
                    if self.context.is_video_file:
                        self.context.update_status(self.tasks_collection, "completed")
                        print(f"[worker {self.context.task_id}] ✅ Video file finished (EOF). Completed.")
                        return
                    time.sleep(0.05)
                    continue

                if use_camera_fps and frame_packet.fps is not None and frame_packet.fps > 0:
                    fps = int(frame_packet.fps)
                    min_interval = 1.0 / max(1, fps)

                try:
                    next_tick, last_status, processed_in_window, skipped_in_window = self.process_one_frame(
                        frame_packet, fps, min_interval, next_tick, last_status, processed_in_window, skipped_in_window
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[worker {self.context.task_id}] ⚠️ Error in process_one_frame (patrol, continuing): {exc}")
                    traceback.print_exc()
                    next_tick = max(next_tick + min_interval, time.time())

            print(f"[worker {self.context.task_id}] 💤 Patrol window ended; going back to sleep")
    
    # -------------------------------------------------------------------------
    # Private: sleep with heartbeat (patrol mode)
    # -------------------------------------------------------------------------

    def sleep_with_heartbeat(self, seconds: int) -> bool:
        """Sleep for given seconds with periodic heartbeat and stop checks. Returns True if should stop."""
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

    # -------------------------------------------------------------------------
    # Private: draw annotations and publish frame to shared_store
    # -------------------------------------------------------------------------

    def draw_and_publish_frame(
        self,
        frame_packet: FramePacket,
        merged_packet: DetectionPacket,
        detections: Dict[str, Any],
        all_matched_indices: List[int],
        rule_match: Optional[RuleMatch] = None,
    ) -> Optional[Any]:
        """Draw bounding boxes and publish processed frame to shared_store."""
        if self.shared_store is None or not self.context.rules:
            if self.shared_store is None:
                print(f"[worker {self.context.task_id}] ⚠️ draw_and_publish_frame: shared_store is None, skip publish")
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
            restricted_zone_coordinates: Optional[List[List[float]]] = None  # Polygon for drawing on frame
            is_line_counting = False
            is_fire_detection = False
            fire_detected = False  # Whether fire is currently detected
            fire_classes = []  # Target classes for fire detection
            is_weapon_detection = False  # Show person + keypoints for pose/skeleton overlay
            is_sleep_detection = False  # Show person + keypoints for pose/skeleton overlay (sleep detection)
            sleep_detection_scenario = None  # For per-box red when sleep confirmed (same box, change color)
            is_wall_climb_detection = False  # Orange = climbing, red = fully above (stays red)
            wall_climb_scenario = None
            is_fall_detection = False  # Red boxes/alert when person fall detected; orange = suspected, red = confirmed
            fall_detection_scenario = None  # For fall_suspected_indices / fall_confirmed_indices and keypoint colors
            fall_red_indices: List[int] = []  # Confirmed fall (red keypoints/boxes); set in fall_detection branch
            fall_suspected_indices: List[int] = []  # Suspected fall (orange keypoints); set in fall_detection branch
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
                        if hasattr(scenario_instance, 'config_obj') and hasattr(scenario_instance.config_obj, 'zone_coordinates'):
                            restricted_zone_coordinates = getattr(scenario_instance.config_obj, 'zone_coordinates', None)
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
                    # Wall climb: orange = climbing, red = fully above (stays red). Reuse zone polygon for drawing.
                    elif scenario_type == 'wall_climb_detection':
                        is_wall_climb_detection = True
                        wall_climb_scenario = scenario_instance
                        if hasattr(scenario_instance, 'config_obj') and hasattr(scenario_instance.config_obj, 'target_class'):
                            target_class = scenario_instance.config_obj.target_class
                        if hasattr(scenario_instance, 'config_obj') and hasattr(scenario_instance.config_obj, 'zone_coordinates'):
                            restricted_zone_coordinates = getattr(scenario_instance.config_obj, 'zone_coordinates', None)
                        if hasattr(scenario_instance, '_state'):
                            state = scenario_instance._state
                            if state.get('red_indices') or state.get('climbing_indices'):
                                zone_violated = True
                    # Face detection: recognized faces (box + name) for overlay
                    elif scenario_type == 'face_detection':
                        if hasattr(scenario_instance, '_state'):
                            face_recognitions = list(scenario_instance._state.get("recognized_faces") or [])
                    # Fall detection: show all persons; orange keypoints = suspected, red = confirmed
                    elif scenario_type == 'fall_detection':
                        is_fall_detection = True
                        fall_detection_scenario = scenario_instance
                        if hasattr(scenario_instance, 'config_obj') and hasattr(scenario_instance.config_obj, 'target_class'):
                            target_class = scenario_instance.config_obj.target_class
                    
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
                                                iou = self.calculate_iou(track_bbox, det_box)
                                                if iou >= 0.3:  # Match threshold
                                                    if idx not in line_crossed_indices:
                                                        line_crossed_indices.append(idx)
                                                    break
            
            # For restricted_zone and wall_climb: draw only target class. Wall climb uses confidence 0.5.
            detections_for_draw = None
            if (is_restricted_zone or is_wall_climb_detection) and target_class:
                kp_src = getattr(merged_packet, "keypoints", None) or []
                draw_conf = 0.5 if is_wall_climb_detection else 0.7
                fd_boxes, fd_classes, fd_scores, _ = self.filter_detections_by_class(
                    target_class, merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    kp_src, confidence_threshold=draw_conf
                )
                detections_for_draw = {"boxes": fd_boxes, "classes": fd_classes, "scores": fd_scores}

            # Fall detection: pass suspected/confirmed indices so keypoints can be drawn orange/red
            # Use original indices from scenario state (they match merged_packet keypoints order)
            if is_fall_detection and fall_detection_scenario and hasattr(fall_detection_scenario, "_state"):
                fall_suspected_orig_for_draw = fall_detection_scenario._state.get("fall_suspected_indices") or []
                fall_confirmed_orig_for_draw = fall_detection_scenario._state.get("fall_confirmed_indices") or []
                # Ensure indices are valid (within keypoints range)
                keypoints_count = len(getattr(merged_packet, "keypoints", []) or [])
                if keypoints_count > 0:
                    fall_suspected_orig_for_draw = [i for i in fall_suspected_orig_for_draw if 0 <= i < keypoints_count]
                    fall_confirmed_orig_for_draw = [i for i in fall_confirmed_orig_for_draw if 0 <= i < keypoints_count]
                detections["fall_suspected_indices"] = fall_suspected_orig_for_draw
                detections["fall_confirmed_indices"] = fall_confirmed_orig_for_draw
            
            # Wall climb detection: pass confirmed indices so keypoints can be drawn red (VLM-confirmed)
            # Use original indices from scenario state (they match merged_packet keypoints order)
            if is_wall_climb_detection and wall_climb_scenario and hasattr(wall_climb_scenario, "_state"):
                wall_climb_confirmed_orig_for_draw = wall_climb_scenario._state.get("confirmed_detection_indices") or []
                # Ensure indices are valid (within keypoints range)
                keypoints_count = len(getattr(merged_packet, "keypoints", []) or [])
                if keypoints_count > 0:
                    wall_climb_confirmed_orig_for_draw = [i for i in wall_climb_confirmed_orig_for_draw if 0 <= i < keypoints_count]
                detections["wall_climb_confirmed_indices"] = wall_climb_confirmed_orig_for_draw
            # Draw annotations based on scenario type
            # For line-based counting (box_count/class_count), frontend handles drawing
            # Backend only draws for pose keypoints or regular bounding boxes
            if detections.get("keypoints"):
                processed_frame = draw_pose_keypoints(frame, detections, self.context.rules)
            elif is_line_counting:
                # For line counting we draw below using track_info from state
                processed_frame = frame.copy()
            else:
                draw_det = detections_for_draw if detections_for_draw else detections
                processed_frame = draw_bounding_boxes(frame, draw_det, self.context.rules)
            
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

            height, width = processed_frame.shape[0], processed_frame.shape[1]

            # For restricted zone: indices into filtered detections that are inside the zone (for per-box red coloring)
            in_zone_indices: List[int] = []
            # Wall climb: confirmed violations (VLM-confirmed) for red keypoints (no bounding boxes)
            wall_climb_confirmed_indices: List[int] = []
            wall_climb_red_indices: List[int] = []  # Deprecated: kept for backward compatibility
            wall_climb_orange_indices: List[int] = []  # Unused; kept for payload compatibility

            # Filter detections based on scenario type (include keypoints for fall_detection/pose)
            keypoints_src = getattr(merged_packet, "keypoints", None) or []
            if is_restricted_zone and target_class:
                # Get scenario's confidence threshold (must match for index mapping to work)
                scenario_conf = 0.5  # Default
                if restricted_zone_scenario and hasattr(restricted_zone_scenario, 'config_obj'):
                    scenario_conf = getattr(restricted_zone_scenario.config_obj, 'confidence_threshold', 0.5)
                
                # Show ALL detections of target class (e.g., all persons) using SAME confidence as scenario
                f_boxes, f_classes, f_scores, f_keypoints = self.filter_detections_by_class(
                    target_class, merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    keypoints_src, confidence_threshold=scenario_conf,
                )
                # Use scenario state in_zone_indices so per-box red is correct even during alert cooldown
                # (when no event is emitted, all_matched_indices is empty but state still has who is in zone)
                state_in_zone = []
                if restricted_zone_scenario and hasattr(restricted_zone_scenario, '_state'):
                    state_in_zone = restricted_zone_scenario._state.get("in_zone_indices") or []
                orig_in_zone = set(state_in_zone) if state_in_zone else set(all_matched_indices)
                
                # Map original indices to filtered indices (only count boxes that pass confidence threshold)
                filtered_idx = 0
                for orig_idx in range(len(merged_packet.boxes)):
                    if orig_idx < len(merged_packet.classes) and orig_idx < len(merged_packet.scores):
                        det_class = merged_packet.classes[orig_idx]
                        det_score = merged_packet.scores[orig_idx]
                        # Only count boxes that pass the SAME confidence threshold as f_boxes
                        if isinstance(det_class, str) and det_class.lower() == target_class and det_score >= scenario_conf:
                            if orig_idx in orig_in_zone:
                                in_zone_indices.append(filtered_idx)
                            filtered_idx += 1
            elif is_wall_climb_detection and target_class:
                # Show ALL detections of target class with keypoints (for red keypoint visualization, no bounding boxes)
                f_boxes, f_classes, f_scores, f_keypoints = self.filter_detections_by_class(
                    target_class, merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    keypoints_src, confidence_threshold=0.5
                )
                # Map confirmed_detection_indices from scenario state (VLM-confirmed violations) to filtered indices
                confirmed_orig = []
                if wall_climb_scenario and hasattr(wall_climb_scenario, '_state'):
                    confirmed_orig = wall_climb_scenario._state.get("confirmed_detection_indices") or []
                # Create mapping: original index -> filtered index
                person_original_indices = []
                for idx, (cls, score) in enumerate(zip(merged_packet.classes, merged_packet.scores)):
                    if isinstance(cls, str) and cls.strip().lower() == target_class and score >= 0.5:
                        person_original_indices.append(idx)
                # Map confirmed original indices to filtered indices
                wall_climb_confirmed_filtered = [i for i, o in enumerate(person_original_indices) if o in confirmed_orig]
                wall_climb_confirmed_indices = wall_climb_confirmed_filtered
                # Legacy: keep empty for backward compatibility (no bounding boxes)
                wall_climb_red_indices = []
            elif is_fire_detection and fire_classes:
                # Show ALL fire-related detections (fire, flame, smoke)
                f_boxes, f_classes, f_scores, f_keypoints = self.filter_detections_by_fire_classes(
                    fire_classes, merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    keypoints_src,
                )
            elif is_weapon_detection or is_sleep_detection:
                # Show ALL person detections with keypoints so UI can draw pose/skeleton
                f_boxes, f_classes, f_scores, f_keypoints = self.filter_detections_by_class(
                    "person", merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    keypoints_src,
                )
            elif is_fall_detection and target_class:
                # Show ALL person detections; map scenario state (original indices) to filtered indices for payload
                f_boxes, f_classes, f_scores, f_keypoints = self.filter_detections_by_class(
                    target_class, merged_packet.boxes, merged_packet.classes, merged_packet.scores,
                    keypoints_src,
                )
                # Same order as filter_detections_by_class (default confidence 0.7)
                person_original_indices = []
                for idx, (cls, score) in enumerate(zip(merged_packet.classes, merged_packet.scores)):
                    if isinstance(cls, str) and cls.strip().lower() == target_class and score >= 0.7:
                        person_original_indices.append(idx)
                fall_suspected_orig = (fall_detection_scenario._state.get("fall_suspected_indices") or []) if fall_detection_scenario and hasattr(fall_detection_scenario, "_state") else []
                fall_confirmed_orig = (fall_detection_scenario._state.get("fall_confirmed_indices") or []) if fall_detection_scenario and hasattr(fall_detection_scenario, "_state") else []
                fall_suspected_filtered = [i for i, o in enumerate(person_original_indices) if o in fall_suspected_orig]
                fall_confirmed_filtered = [i for i, o in enumerate(person_original_indices) if o in fall_confirmed_orig]
                fall_suspected_indices = fall_suspected_filtered
                fall_red_indices = fall_confirmed_filtered
            elif is_line_counting and target_class:
                # Show ALL detections of target class for line counting
                f_boxes, f_classes, f_scores, f_keypoints = self.filter_detections_by_class(
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
                f_boxes, f_classes, f_scores, f_keypoints = self.filter_detections_by_indices(
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
                        if self.calculate_iou(box, eb) >= 0.3:
                            sleep_confirmed_indices.append(i)
                            break

            # Fall detection: red = confirmed fall (per-person); set above in fall_detection branch; legacy fallback below
            if is_fall_detection and rule_match and getattr(rule_match, "event_type", None) == "fall_detected" and f_boxes and not fall_red_indices:
                fall_red_indices = list(range(len(f_boxes)))

            # Draw restricted zone polygon, zone line, and red boxes for in-zone detections (on processed frame)
            draw_zone_polygon(processed_frame, restricted_zone_coordinates, width, height)
            draw_zone_line(processed_frame, line_zone, width, height)
            draw_boxes_in_zone_red(processed_frame, f_boxes, f_classes, f_scores, in_zone_indices)
            # Wall climb: NO bounding boxes - only red keypoints are drawn after VLM confirmation
            # Bounding boxes removed for wall climb detection as per requirement
            # Fall detection: NO bounding boxes - only keypoints are drawn (orange/red)
            # Bounding boxes removed for fall detection as per requirement

            # Collect scenario overlays (e.g., face boxes, loom ROI) before encoding so we can draw them on the frame
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
                            overlay_data_to_dict_list(overlay_data)
                        )
                        # Loom machine state: draw polygons directly on frame
                        if scenario_type == "loom_machine_state" and isinstance(overlay_data, dict):
                            polygons = overlay_data.get("polygons", [])
                            colors = overlay_data.get("colors", [])
                            if polygons and colors:
                                draw_loom_machine_polygons(
                                    processed_frame,
                                    polygons,
                                    colors,
                                    width,
                                    height,
                                    thickness=3,
                                    fill_alpha=0.3
                                )
                        # Face detection: use same overlay for face_recognitions so UI and stream both get boxes
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
            # Draw face recognition boxes on the frame so they appear in the agent stream (JPEG)
            draw_face_recognitions(processed_frame, face_recognitions)
            frame_bytes = processed_frame.tobytes()

            # Zone for UI overlay (restricted_zone and wall_climb both use polygon)
            zone_for_payload = None
            if restricted_zone_coordinates and len(restricted_zone_coordinates) >= 3:
                zone_for_payload = {"type": "polygon", "coordinates": restricted_zone_coordinates}

            # Publish to shared_store (key must be string so overlay/ws can read by agent_id from URL)
            store_key = str(self.context.agent_id)
            payload = {
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
                "zone": zone_for_payload,  # Polygon zone for UI (restricted_zone / wall_climb)
                "zone_violated": zone_violated,  # Zone violation status
                "line_zone": line_zone,  # Line zone data for visualization
                "line_crossed": len(line_crossed_indices) > 0,  # Whether any object crossed the line
                "line_crossed_indices": line_crossed_indices,  # Indices of filtered detections that crossed the line
                "track_info": track_info,  # Track information (center points, track IDs) for visualization
                "fire_detected": fire_detected,  # Fire detection status (for red bounding boxes)
                "in_zone_indices": in_zone_indices,  # Restricted zone: indices of filtered detections inside zone (red box only these)
                "sleep_confirmed_indices": sleep_confirmed_indices,  # Sleep: same person box, red when VLM confirmed
                "wall_climb_confirmed_indices": wall_climb_confirmed_indices,  # Wall climb: VLM-confirmed violations (red keypoints only, no bounding boxes)
                "wall_climb_red_indices": wall_climb_red_indices,  # Deprecated: kept for backward compatibility
                "wall_climb_orange_indices": wall_climb_orange_indices,  # Unused; kept for UI compatibility
                "fall_detected": bool(is_fall_detection and fall_red_indices),  # Fall: show red alert on UI
                "fall_red_indices": fall_red_indices,  # Fall: confirmed fall (red keypoints/boxes)
                "fall_suspected_indices": fall_suspected_indices,  # Fall: suspected fall (orange keypoints)
                "face_recognitions": face_recognitions,  # Face detection: [{ "box": [x1,y1,x2,y2], "name": "..." }]
            }
            try:
                self.shared_store[store_key] = payload
                # Debug: log every 30th frame (and first 3) so we see writes without flooding
                idx = frame_packet.frame_index
                if idx <= 3 or (idx % 30 == 0):
                    print(f"[worker {self.context.task_id}] 📤 shared_store WRITE ok | store_key={store_key!r} frame_index={idx}")
            except Exception as write_exc:  # noqa: BLE001
                print(f"[worker {self.context.task_id}] ⚠️ shared_store write failed (stream may be stale): {write_exc}")
            return processed_frame
        except Exception as exc:  # noqa: BLE001
            print(f"[worker {self.context.task_id}] ⚠️  Error processing frame for stream: {exc}")
            return None
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
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

    # -------------------------------------------------------------------------
    # Private: filter detections by indices, class, or fire classes
    # -------------------------------------------------------------------------

    def filter_detections_by_indices(
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
    
    def filter_detections_by_class(
        self,
        target_class: str,
        boxes: List[List[float]],
        classes: List[str],
        scores: List[float],
        keypoints: Optional[List[List[List[float]]]] = None,
        confidence_threshold: Optional[float] = None,
    ) -> tuple[List[List[float]], List[str], List[float], List[List[List[float]]]]:
        """Filter detections to only those matching target class (e.g., 'person') with confidence >= threshold.
        When keypoints is provided (e.g. for fall_detection/pose), returns aligned f_keypoints.
        confidence_threshold: default 0.7; use 0.5 for wall_climb_detection."""
        f_boxes: List[List[float]] = []
        f_classes: List[str] = []
        f_scores: List[float] = []
        f_keypoints: List[List[List[float]]] = []
        target_lower = target_class.lower()
        conf = 0.7 if confidence_threshold is None else float(confidence_threshold)
        kp_list = keypoints if keypoints else []

        # Debug: Log all detected classes
        if len(classes) > 0:
            unique_classes = set(cls.lower() if isinstance(cls, str) else str(cls) for cls in classes)
            print(f"[worker {self.context.task_id}] 🔍 YOLO detected classes: {unique_classes} (total: {len(classes)} objects)")
        
        filtered_count = 0
        for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if isinstance(cls, str) and cls.lower() == target_lower:
                # Apply confidence threshold
                if score >= conf:
                    f_boxes.append(box)
                    f_classes.append(cls)
                    f_scores.append(score)
                    if idx < len(kp_list):
                        f_keypoints.append(kp_list[idx])
                else:
                    filtered_count += 1
        
        # Debug: Log filtering results
        if len(f_boxes) > 0:
            print(f"[worker {self.context.task_id}] ✅ Found {len(f_boxes)} '{target_class}' detections (confidence >= {conf})")
        if filtered_count > 0:
            print(f"[worker {self.context.task_id}] 🔽 Filtered out {filtered_count} low-confidence '{target_class}' detections (< {conf})")
        
        return f_boxes, f_classes, f_scores, f_keypoints
    
    def filter_detections_by_fire_classes(
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

    # -------------------------------------------------------------------------
    # Private: handle rule match event (session manager, notifications)
    # -------------------------------------------------------------------------

    def handle_event(
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
        event_type = getattr(rule_match, "event_type", None) or ""
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
                    fps=fps,
                    event_type=event_type,
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
                    fps=fps,
                    event_type=event_type,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[worker {self.context.task_id}] ⚠️  Error creating/handling event frame: {exc}")
