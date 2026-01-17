"""
Agent main loop
---------------

Implements the per-agent processing with hybrid behavior:
- Load models
- Open source
- Modes:
  - continuous: read and process frames at agent FPS indefinitely
  - patrol: sleep for interval, then process frames for a short window at agent FPS, repeat
- Merge detections and apply rule engine
- Print alerts, send heartbeat
- Stop on end_at/stop_requested/file end
"""
import time
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from bson import ObjectId
from bson.errors import InvalidId
try:
    import cv2  # type: ignore
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore[assignment]

# Allow running when imported without installed package by ensuring project root is on sys.path
if __package__ is None or __package__ == "":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

from app.utils.db import get_collection
from app.utils.datetime_utils import now, utc_now, parse_iso, mongo_datetime_to_app_timezone
from app.utils.event_notifier import send_event_to_backend_sync
from app.utils.event_session_manager import get_event_session_manager
from app.processing.yolo_model.yolo_utils import init_yolo_model, check_event_match
from app.processing.rule_engine.engine import evaluate_rules
from app.processing.rule_engine.registry import rules_registry
from app.processing.worker.video_io import open_video_capture
from app.processing.worker.detections import extract_detections_from_result
from app.processing.worker.detections import extract_keypoints_from_result
from app.processing.worker.frame_hub import reconstruct_frame
from app.processing.worker.frame_processor import draw_bounding_boxes
from app.processing.worker.frame_processor import draw_pose_keypoints
import numpy as np


def _filter_detections_by_indices(
    indices: List[int],
    boxes: List[List[float]],
    classes: List[str],
    scores: List[float],
) -> tuple[List[List[float]], List[str], List[float]]:
    """
    Filter detections to only those at the specified indices.
    Generic helper that works with any rule's matched_detection_indices.
    """
    if not indices:
        return [], [], []
    f_boxes: List[List[float]] = []
    f_classes: List[str] = []
    f_scores: List[float] = []
    for idx in indices:
        if 0 <= idx < len(boxes) and idx < len(classes) and idx < len(scores):
            f_boxes.append(boxes[idx])
            f_classes.append(classes[idx])
            f_scores.append(scores[idx])
    return f_boxes, f_classes, f_scores


def sleep_with_heartbeat(tasks_collection, task_id: str, seconds: int) -> bool:
    """
    Sleep for 'seconds' but wake up every 1 second to:
    - send heartbeat
    - check stop/end
    Returns True if a stop/end condition is met (caller should exit), False otherwise.
    """
    end_time = time.time() + max(0, seconds)
    while time.time() < end_time:
        # Heartbeat
        tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"updated_at": utc_now()}})
        # Stop / end checks - support both old and new field names
        task_document = tasks_collection.find_one({"_id": ObjectId(task_id)}, projection={"stop_requested": 1, "end_time": 1, "end_at": 1})
        # If task document was deleted, stop
        if not task_document:
            print(f"[worker {task_id}] ⏹️ Stopping (task deleted)")
            return True
        stop_requested = bool(task_document.get("stop_requested")) if task_document else False
        # Support both old and new field names
        end_at_value = task_document.get("end_time") or task_document.get("end_at") if task_document else None
        # Handle both datetime objects (from MongoDB) and strings
        if isinstance(end_at_value, datetime):
            # MongoDB datetime objects are UTC (naive), convert to app timezone
            end_at_dt = mongo_datetime_to_app_timezone(end_at_value)
        else:
            end_at_dt = parse_iso(end_at_value) if end_at_value else None
        current_time = now()
        if stop_requested or (end_at_dt and current_time >= end_at_dt):
            status = "completed" if end_at_dt and current_time >= end_at_dt else "cancelled"
            tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": status, "stopped_at": utc_now(), "updated_at": utc_now()}})
            print(f"[worker {task_id}] ⏹️ Stopping (status={status}, end_at={end_at_dt}, now={current_time})")
            return True
        time.sleep(1)
    return False


def format_video_time_ms(milliseconds: float) -> str:
    """
    Convert milliseconds to H:MM:SS.mmm string.
    """
    if milliseconds < 0:
        milliseconds = 0
    total_seconds, milliseconds_remainder = divmod(int(milliseconds), 1000)
    hours, remaining_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{milliseconds_remainder:03d}"


def get_video_time_ms(video_capture, frame_index: int, fps: int) -> float:
    """
    Read video time from OpenCV if available, else estimate using frame_index/fps.
    """
    position_ms = -1.0
    if cv2 is not None and video_capture is not None:
        try:
            position_ms = float(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        except Exception:
            position_ms = -1.0
    if position_ms is None or position_ms <= 0:
        if fps > 0:
            return (frame_index / float(fps)) * 1000.0
        return 0.0
    return position_ms


def run_task_worker(task_id: str, shared_store: Optional["Dict[str, Any]"] = None) -> None:
    """
    Standalone agent main loop.
    """
    tasks_collection = get_collection()
    # Query by _id (MongoDB's native field) with ObjectId conversion
    try:
        task = tasks_collection.find_one({"_id": ObjectId(task_id)})
    except (InvalidId, ValueError, TypeError):
        # Fallback: try querying by "id" field if ObjectId conversion fails
        task = tasks_collection.find_one({"id": task_id})
    
    if not task:
        print(f"[worker {task_id}] ❌ Task not found. Exiting.")
        return

    # Support both old and new field names
    agent_name = task.get("name") or task.get("task_name") or f"agent-{task_id}"
    fps = int(task.get("fps", 5))
    
    # Handle model: new format uses single "model" string, old format uses "model_ids" array
    model_value = task.get("model")
    if model_value:
        # New format: single model string
        if isinstance(model_value, str):
            model_ids = [model_value]
        elif isinstance(model_value, list):
            model_ids = model_value
        else:
            model_ids = []
    else:
        # Old format: model_ids array
        model_ids = task.get("model_ids", []) or []
    
    run_mode = (task.get("run_mode") or "continuous").strip().lower()  # "continuous" | "patrol"
    # Handle None values explicitly (MongoDB null values)
    interval_minutes = int(task.get("interval_minutes") or 5)  # patrol sleep interval
    check_duration_seconds = int(task.get("check_duration_seconds") or 10)  # patrol detection window

    print(f"[worker {task_id}] ▶️ Starting '{agent_name}' | mode={run_mode} fps={fps} models={model_ids}")

    # Load models
    models = []
    for model_id in model_ids:
        model_instance = init_yolo_model({"yolo_model_path": model_id})
        if model_instance is not None:
            models.append(model_instance)
        else:
            print(f"[worker {task_id}] ⚠️ Failed to load model: {model_id}")
    if not models:
        print(f"[worker {task_id}] ❌ No models loaded. Exiting.")
        return

    # Decide if using centralized hub (stream) or direct file capture
  
    camera_id = (task.get("camera_id") or "").strip()
    use_hub = bool(camera_id and shared_store is not None)

    # Open source (file fallback only)
    video_capture = None
    if not use_hub:
        video_capture, _capture_source = open_video_capture(task)
        if video_capture is None:
            # For file tasks with no capture, mark completed
            if (task.get("source_type") == "file") or (not task.get("source_uri") and bool(task.get("file_path"))):
                tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": "completed", "stopped_at": utc_now(), "updated_at": utc_now()}})
                print(f"[worker {task_id}] ⚠️ Could not open file. Marked completed.")
            return

    # Decide if this source should be treated as a file (finite) or stream (infinite)
    source_is_file = (task.get("source_type") == "file") or (not task.get("source_uri") and bool(task.get("file_path")))

    # Load rules ONCE at startup (do not refetch per frame)
    loaded_rules: List[Dict[str, any]] = task.get("rules") or []

    try:
        # per-agent in-memory rule state (indexed by rule index)
        rule_state: Dict[int, Dict[str, any]] = {}

        if run_mode == "continuous":
            # Process frames at FPS continuously
            min_interval = 1.0 / max(1, fps)
            next_tick = time.time()
            frame_index = 0
            # sampling diagnostics
            last_status = time.time()
            processed_in_window = 0
            skipped_in_window = 0
            last_seen_hub_index: Optional[int] = None
            while True:
                # stop/end checks - support both old and new field names
                
                task_document = tasks_collection.find_one({"_id": ObjectId(task_id)}, projection={"stop_requested": 1, "end_time": 1, "end_at": 1})
                # If task document deleted, stop immediately
                if not task_document:
                    print(f"[worker {task_id}] ⏹️ Stopping (task deleted)")
                    return
                stop_requested = bool(task_document.get("stop_requested")) if task_document else False
                # Support both old and new field names
                end_at_value = task_document.get("end_time") or task_document.get("end_at") if task_document else None
                # Handle both datetime objects (from MongoDB) and strings
                if isinstance(end_at_value, datetime):
                    # MongoDB datetime objects are UTC (naive), convert to app timezone
                    end_at_dt = mongo_datetime_to_app_timezone(end_at_value)
                else:
                    end_at_dt = parse_iso(end_at_value) if end_at_value else None
                current_time = now()
                if stop_requested or (end_at_dt and current_time >= end_at_dt):
                    status = "completed" if end_at_dt and current_time >= end_at_dt else "cancelled"
                    try:
                        tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": status, "stopped_at": utc_now(), "updated_at": utc_now()}})
                    except (InvalidId, ValueError, TypeError):
                        # Fallback: try updating by "id" field if ObjectId conversion fails
                        tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": status, "stopped_at": utc_now(), "updated_at": utc_now()}})
                    print(f"[worker {task_id}] ⏹️ Stopping (status={status}, end_at={end_at_dt}, now={current_time})")
                    return

                # FPS pacing
                # FPS pacing: control how often we process frames so we match the target FPS.
                # 'next_tick' is the theoretical time (in seconds since epoch) when we should process the next frame.
                now_timestamp = time.time()
                if now_timestamp < next_tick:
                    # If we're ahead of schedule, sleep until the next scheduled frame time.
                    time.sleep(max(0.0, next_tick - now_timestamp))
                # After processing, compute the next frame's scheduled time.
                # This ensures steady pacing and prevents drift: 
                # - next_tick is increased by min_interval (frame period)
                # - But if we're running late (e.g., heavy computation), catch up to the current time
                next_tick = max(next_tick + min_interval, time.time())

                # Acquire next frame
                if use_hub:
                    entry = shared_store.get(camera_id, {}) if shared_store is not None else {}
                    frame = reconstruct_frame(entry)
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    # sampling metrics
                    hub_index = int(entry.get("frame_index", 0)) if isinstance(entry, dict) else 0
                    if last_seen_hub_index is not None and hub_index > last_seen_hub_index + 1:
                        skipped_in_window += (hub_index - last_seen_hub_index - 1)
                    last_seen_hub_index = hub_index
                else:
                    is_frame_read, frame = video_capture.read()  # type: ignore[union-attr]
                    if not is_frame_read:
                        # File: end reached → complete. Stream: brief retry.
                        if source_is_file:
                            try:
                                tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": "completed", "stopped_at": utc_now(), "updated_at": utc_now()}})
                            except (InvalidId, ValueError, TypeError):
                                # Fallback: try updating by "id" field if ObjectId conversion fails
                                tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": "completed", "stopped_at": utc_now(), "updated_at": utc_now()}})
                            print(f"[worker {task_id}] ✅ File ended. Marked completed.")
                            return
                        else:
                            time.sleep(0.3)
                            continue

                frame_index += 1
                processed_in_window += 1
                # status once per second
                if (time.time() - last_status) >= 1.0:
                    base_fps = 0.0
                    hub_index_status = last_seen_hub_index or 0
                    if use_hub and isinstance(entry, dict):
                        try:
                            base_fps = float(entry.get("base_fps", 0.0))
                        except Exception:
                            base_fps = 0.0
                    print(f"[worker {task_id}] ⏱️ sampling agent_fps={fps} base_fps={base_fps} processed={processed_in_window}/s skipped={skipped_in_window} hub_frame_index={hub_index_status} camera_id={camera_id}")
                    processed_in_window = 0
                    skipped_in_window = 0
                    last_status = time.time()

                # Run YOLO on this frame for each model, merge detections
                merged_boxes: List[List[float]] = []
                merged_classes: List[str] = []
                merged_scores: List[float] = []
                merged_keypoints: List[List[List[float]]] = []
                
                for model in models:
                    try:
                        results = model(frame, verbose=False)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[worker {task_id}] ⚠️ YOLO error: {exc}")
                        continue
                    if results:
                        first_result = results[0]
                        boxes, classes, scores = extract_detections_from_result(first_result)
                        merged_boxes.extend(boxes)
                        merged_classes.extend(classes)
                        merged_scores.extend(scores)
                        keypoints = extract_keypoints_from_result(first_result)
                        if keypoints:
                            merged_keypoints.extend(keypoints)

                # Build detections payload for rule engine
                detections = {
                    "classes": merged_classes,
                    "scores": merged_scores,
                    "boxes": merged_boxes,
                    "keypoints": merged_keypoints,
                    "ts": now(),
                }
                
                # Debug: Log keypoint extraction (first frame and every 30 frames)
                if frame_index == 1 or frame_index % 30 == 0:
                    if merged_keypoints:
                        print(f"[worker {task_id}] ✅ Keypoints extracted: {len(merged_keypoints)} person(s) with pose data")
                    else:
                        print(f"[worker {task_id}] ⚠️ No keypoints extracted! Check if model is a pose model (e.g., yolov8n-pose.pt)")
                
                # Draw bounding boxes and publish processed frame for agent stream
                processed_frame = None
                # Support both old and new field names
                agent_id = task.get("id") or task.get("agent_id") or task_id
                
                if shared_store is not None and loaded_rules:
                    try:
                        # Draw pose keypoints if available; otherwise draw bounding boxes
                        if detections.get("keypoints"):
                            processed_frame = draw_pose_keypoints(frame.copy(), detections, loaded_rules)
                        else:
                            processed_frame = draw_bounding_boxes(frame.copy(), detections, loaded_rules)
                        
                        # Convert processed frame to bytes
                        frame_bytes = processed_frame.tobytes()
                        height, width = processed_frame.shape[0], processed_frame.shape[1]
                        
                        # Get frame metadata from hub entry if available
                        hub_frame_index = entry.get("frame_index", frame_index) if use_hub and isinstance(entry, dict) else frame_index
                        camera_fps = entry.get("camera_fps") if use_hub and isinstance(entry, dict) else None
                        actual_fps = entry.get("actual_fps") if use_hub and isinstance(entry, dict) else fps
                        
                        # Filter overlay detections using matched_detection_indices from rule evaluation
                        # (generic: works with any rule type, no hardcoded rule logic)
                        if all_matched_indices:
                            # Deduplicate indices
                            unique_indices = sorted(set(all_matched_indices))
                            f_boxes, f_classes, f_scores = _filter_detections_by_indices(
                                unique_indices, merged_boxes, merged_classes, merged_scores
                            )
                        else:
                            # No rule-matched detections: show empty overlay (agent rules don't match current frame)
                            f_boxes, f_classes, f_scores = [], [], []

                        # Publish processed frame to agent-specific shared_store key
                        shared_store[agent_id] = {
                            "shape": (height, width, 3),
                            "dtype": "uint8",
                            "frame_index": hub_frame_index,
                            "ts_monotonic": time.time(),
                            "camera_fps": camera_fps,
                            "actual_fps": actual_fps,
                            "bytes": frame_bytes,
                            "agent_id": agent_id,
                            "task_name": agent_name,
                            # Overlay metadata (Option A): filtered by rule-matched detection indices
                            "detections": {
                                "boxes": f_boxes,
                                "classes": f_classes,
                                "scores": f_scores,
                            },
                            "rules": loaded_rules,
                            "camera_id": camera_id,
                        }
                    except Exception as exc:  # noqa: BLE001
                        # Don't fail the worker if frame processing fails
                        print(f"[worker {task_id}] ⚠️  Error processing frame for stream: {exc}")

                # Prefer new rule engine if rules are provided; else fallback
                event = None
                all_matched_indices: List[int] = []
                if loaded_rules:
                    # Debug: log rules and detections (first frame only, then every 30 frames)
                    if frame_index == 1 or frame_index % 30 == 0:
                        print(f"[worker {task_id}] 🔍 Rules: {loaded_rules}")
                        print(f"[worker {task_id}] 🔍 Detected classes: {merged_classes[:5]}")  # First 5 only
                    
                    # Evaluate all agent rules to collect matched_detection_indices for overlay
                    # (even if no event triggers, we want to show rule-relevant boxes)
                    for rule_idx, rule in enumerate(loaded_rules or []):
                        rule_type = str(rule.get("type", "")).strip().lower()
                        handler = rules_registry.get(rule_type)
                        if handler:
                            try:
                                # Evaluate this agent's rule to get matched indices
                                rule_result = handler(rule, detections, task, rule_state.setdefault(rule_idx, {}), detections["ts"])
                                if rule_result and isinstance(rule_result, dict):
                                    idx_list = rule_result.get("matched_detection_indices")
                                    if isinstance(idx_list, list):
                                        all_matched_indices.extend(idx_list)
                                    # If this rule triggered an event and we don't have one yet, use it
                                    if not event and rule_result.get("label"):
                                        event = rule_result
                                        event.setdefault("rule_index", rule_idx)
                            except Exception:
                                pass  # ignore per-rule errors for overlay collection
                    
                    # If no event was found from individual evaluation, try evaluate_rules for event detection
                    # (this handles duration_seconds and other event-specific logic)
                    if not event:
                        event = evaluate_rules(loaded_rules, detections, task, rule_state, detections["ts"])
                # else:
                    # compatibility path
                    # label = check_event_match(task, merged_classes)
                    # event = {"label": label} if label else None

                video_ms = get_video_time_ms(video_capture, frame_index, fps) if not use_hub else (frame_index / float(max(1, fps))) * 1000.0
                video_ts = format_video_time_ms(video_ms)

                if event and event.get("label"):
                    event_label = str(event["label"]).strip()
                    print(f"[worker {task_id}] 🔔 {event_label} | agent='{agent_name}' | video_time={video_ts}")
                    
                    # Handle event through session manager (with runtime video splitting)
                    print(f"[worker {task_id}] 📋 Event handling: processed_frame={'exists' if processed_frame is not None else 'None'}, loaded_rules={'exists' if loaded_rules else 'None'}, shared_store={'exists' if shared_store is not None else 'None'}")
                    
                    if processed_frame is not None:
                        try:
                            print(f"[worker {task_id}] 📤 Calling handle_event_frame with existing processed_frame")
                            session_manager = get_event_session_manager()
                            session_manager.handle_event_frame(
                                agent_id=agent_id,
                                rule_index=event.get("rule_index", 0),
                                event_label=event_label,
                                frame=processed_frame,
                                camera_id=camera_id,
                                agent_name=agent_name,
                                detections=detections,
                                video_timestamp=video_ts,
                                fps=fps
                            )
                            print(f"[worker {task_id}] ✅ handle_event_frame completed successfully")
                        except Exception as exc:  # noqa: BLE001
                            print(f"[worker {task_id}] ⚠️  Error handling event frame: {exc}")
                            import traceback
                            print(f"[worker {task_id}] Traceback: {traceback.format_exc()}")
                    elif loaded_rules:
                        # If processed_frame wasn't created but we have rules, create it now for event notification
                        try:
                            print(f"[worker {task_id}] 🎨 Creating processed_frame for event notification")
                            # Draw pose keypoints if available; otherwise draw bounding boxes
                            if detections.get("keypoints"):
                                processed_frame = draw_pose_keypoints(frame.copy(), detections, loaded_rules)
                            else:
                                processed_frame = draw_bounding_boxes(frame.copy(), detections, loaded_rules)
                            print(f"[worker {task_id}] 📤 Calling handle_event_frame with newly created processed_frame")
                            session_manager = get_event_session_manager()
                            session_manager.handle_event_frame(
                                agent_id=agent_id,
                                rule_index=event.get("rule_index", 0),
                                event_label=event_label,
                                frame=processed_frame,
                                camera_id=camera_id,
                                agent_name=agent_name,
                                detections=detections,
                                video_timestamp=video_ts,
                                fps=fps
                            )
                            print(f"[worker {task_id}] ✅ handle_event_frame completed successfully")
                        except Exception as exc:  # noqa: BLE001
                            print(f"[worker {task_id}] ⚠️  Error creating/handling event frame: {exc}")
                            import traceback
                            print(f"[worker {task_id}] Traceback: {traceback.format_exc()}")
                    else:
                        print(f"[worker {task_id}] ⚠️  Cannot send event: processed_frame is None and loaded_rules is empty")
                else:
                    pass
                    # print(f"[worker {task_id}] ℹ️ No rule match | agent='{agent_name}' | video_time={video_ts}")

                # Heartbeat
                try:
                    tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"updated_at": utc_now()}})
                except (InvalidId, ValueError, TypeError):
                    # Fallback: try updating by "id" field if ObjectId conversion fails
                    tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"updated_at": utc_now()}})
        else:
            # Patrol mode: sleep interval, then process frames for a short window at FPS, repeat
            interval_seconds = max(0, int(interval_minutes) * 60)
            window_seconds = max(1, int(check_duration_seconds))
            print(f"[worker {task_id}] 💤 Patrol mode | sleep={interval_seconds}s window={window_seconds}s fps={fps}")
            while True:
                # Sleep with periodic heartbeat and stop checks
                if sleep_with_heartbeat(tasks_collection, task_id, interval_seconds):
                    return

                # Detection window
                window_end = time.time() + window_seconds
                min_interval = 1.0 / max(1, fps)
                next_tick = time.time()
                print(f"[worker {task_id}] 🔎 Patrol window started ({window_seconds}s)")
                # Reset per-window rule state for patrol windows
                rule_state = {}
                frame_index = 0
                # patrol diagnostics
                last_status = time.time()
                processed_in_window = 0
                skipped_in_window = 0
                last_seen_hub_index = None
                while time.time() < window_end:
                    # stop/end checks inside window - support both old and new field names
                    task_document = tasks_collection.find_one({"_id": ObjectId(task_id)}, projection={"stop_requested": 1, "end_time": 1, "end_at": 1})
                    # If task document deleted, stop immediately
                    if not task_document:
                        print(f"[worker {task_id}] ⏹️ Stopping (task deleted)")
                        return
                    stop_requested = bool(task_document.get("stop_requested")) if task_document else False
                    # Support both old and new field names
                    end_at_value = task_document.get("end_time") or task_document.get("end_at") if task_document else None
                    # Handle both datetime objects (from MongoDB) and strings
                    if isinstance(end_at_value, datetime):
                        end_at_dt = end_at_value
                        # Ensure timezone-aware (MongoDB dates might be naive)
                        if end_at_dt.tzinfo is None:
                            # Use centralized datetime utility to get app timezone
                            from app.utils.datetime_utils import _get_app_timezone
                            end_at_dt = end_at_dt.replace(tzinfo=_get_app_timezone())
                    else:
                        end_at_dt = parse_iso(end_at_value) if end_at_value else None
                    current_time = now()
                    if stop_requested or (end_at_dt and current_time >= end_at_dt):
                        status = "completed" if end_at_dt and current_time >= end_at_dt else "cancelled"
                        try:
                            tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": status, "stopped_at": utc_now(), "updated_at": utc_now()}})
                        except (InvalidId, ValueError, TypeError):
                            # Fallback: try updating by "id" field if ObjectId conversion fails
                            tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": status, "stopped_at": utc_now(), "updated_at": utc_now()}})
                        print(f"[worker {task_id}] ⏹️ Stopping (status={status}, end_at={end_at_dt}, now={current_time})")
                        return

                    # FPS pacing
                    now_timestamp = time.time()
                    if now_timestamp < next_tick:
                        time.sleep(max(0.0, next_tick - now_timestamp))
                    next_tick = max(next_tick + min_interval, time.time())

                    # Acquire next frame
                    if use_hub:
                        entry = shared_store.get(camera_id, {}) if shared_store is not None else {}
                        frame = reconstruct_frame(entry)
                        if frame is None:
                            time.sleep(0.05)
                            continue
                        # sampling metrics
                        hub_index = int(entry.get("frame_index", 0)) if isinstance(entry, dict) else 0
                        if last_seen_hub_index is not None and hub_index > last_seen_hub_index + 1:
                            skipped_in_window += (hub_index - last_seen_hub_index - 1)
                        last_seen_hub_index = hub_index
                    else:
                        is_frame_read, frame = video_capture.read()  # type: ignore[union-attr]
                        if not is_frame_read:
                            if source_is_file:
                                try:
                                    tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": "completed", "stopped_at": utc_now(), "updated_at": utc_now()}})
                                except (InvalidId, ValueError, TypeError):
                                    # Fallback: try updating by "id" field if ObjectId conversion fails
                                    tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": "completed", "stopped_at": utc_now(), "updated_at": utc_now()}})
                                print(f"[worker {task_id}] ✅ File ended. Marked completed.")
                                return
                            else:
                                time.sleep(0.3)
                                continue

                    frame_index += 1
                    processed_in_window += 1
                    if (time.time() - last_status) >= 1.0:
                        base_fps = 0.0
                        hub_index_status = last_seen_hub_index or 0
                        if use_hub and isinstance(entry, dict):
                            try:
                                base_fps = float(entry.get("base_fps", 0.0))
                            except Exception:
                                base_fps = 0.0
                        print(f"[worker {task_id}] ⏱️ sampling(agent patrol) agent_fps={fps} base_fps={base_fps} processed={processed_in_window}/s skipped={skipped_in_window} hub_frame_index={hub_index_status} camera_id={camera_id}")
                        processed_in_window = 0
                        skipped_in_window = 0
                        last_status = time.time()
                    merged_boxes: List[List[float]] = []
                    merged_classes: List[str] = []
                    merged_scores: List[float] = []
                    merged_keypoints: List[List[List[float]]] = []
                    
                    for model in models:
                        try:
                            results = model(frame, verbose=False)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[worker {task_id}] ⚠️ YOLO error: {exc}")
                            continue
                        if results:
                            first_result = results[0]
                            boxes, classes, scores = extract_detections_from_result(first_result)
                            merged_boxes.extend(boxes)
                            merged_classes.extend(classes)
                            merged_scores.extend(scores)
                            keypoints = extract_keypoints_from_result(first_result)
                            if keypoints:
                                merged_keypoints.extend(keypoints)

                    detections = {
                        "classes": merged_classes,
                        "scores": merged_scores,
                        "boxes": merged_boxes,
                        "keypoints": merged_keypoints,
                        "ts": now(),
                    }
                    
                    # Evaluate rules first to get matched_detection_indices for overlay filtering
                    event = None
                    all_matched_indices_patrol: List[int] = []
                    if loaded_rules:
                        # Evaluate all agent rules to collect matched_detection_indices for overlay
                        # (even if no event triggers, we want to show rule-relevant boxes)
                        for rule_idx, rule in enumerate(loaded_rules or []):
                            rule_type = str(rule.get("type", "")).strip().lower()
                            handler = rules_registry.get(rule_type)
                            if handler:
                                try:
                                    # Evaluate this agent's rule to get matched indices
                                    rule_result = handler(rule, detections, task, rule_state.setdefault(rule_idx, {}), detections["ts"])
                                    if rule_result and isinstance(rule_result, dict):
                                        idx_list = rule_result.get("matched_detection_indices")
                                        if isinstance(idx_list, list):
                                            all_matched_indices_patrol.extend(idx_list)
                                        # If this rule triggered an event and we don't have one yet, use it
                                        if not event and rule_result.get("label"):
                                            event = rule_result
                                            event.setdefault("rule_index", rule_idx)
                                except Exception:
                                    pass  # ignore per-rule errors for overlay collection
                        
                        # If no event was found from individual evaluation, try evaluate_rules for event detection
                        # (this handles duration_seconds and other event-specific logic)
                        if not event:
                            event = evaluate_rules(loaded_rules, detections, task, rule_state, detections["ts"])
                    
                    # Draw bounding boxes and publish processed frame for agent stream (patrol mode)
                    processed_frame = None
                    # Support both old and new field names
                    agent_id = task.get("id") or task.get("agent_id") or task_id
                    
                    if shared_store is not None and loaded_rules:
                        try:
                            # Draw pose keypoints if available; otherwise draw bounding boxes
                            if detections.get("keypoints"):
                                processed_frame = draw_pose_keypoints(frame.copy(), detections, loaded_rules)
                            else:
                                processed_frame = draw_bounding_boxes(frame.copy(), detections, loaded_rules)
                            frame_bytes = processed_frame.tobytes()
                            height, width = processed_frame.shape[0], processed_frame.shape[1]
                            hub_frame_index = entry.get("frame_index", frame_index) if use_hub and isinstance(entry, dict) else frame_index
                            camera_fps = entry.get("camera_fps") if use_hub and isinstance(entry, dict) else None
                            actual_fps = entry.get("actual_fps") if use_hub and isinstance(entry, dict) else fps
                            
                            # Filter overlay detections using matched_detection_indices (generic, rule-agnostic)
                            if all_matched_indices_patrol:
                                unique_indices = sorted(set(all_matched_indices_patrol))
                                f_boxes, f_classes, f_scores = _filter_detections_by_indices(
                                    unique_indices, merged_boxes, merged_classes, merged_scores
                                )
                            else:
                                f_boxes, f_classes, f_scores = [], [], []

                            shared_store[agent_id] = {
                                "shape": (height, width, 3),
                                "dtype": "uint8",
                                "frame_index": hub_frame_index,
                                "ts_monotonic": time.time(),
                                "camera_fps": camera_fps,
                                "actual_fps": actual_fps,
                                "bytes": frame_bytes,
                                "agent_id": agent_id,
                                "task_name": agent_name,
                                # Overlay metadata (Option A): filtered by rule-matched detection indices
                                "detections": {
                                    "boxes": f_boxes,
                                    "classes": f_classes,
                                    "scores": f_scores,
                                },
                                "rules": loaded_rules,
                                "camera_id": camera_id,
                            }
                        except Exception as exc:  # noqa: BLE001
                            print(f"[worker {task_id}] ⚠️  Error processing frame for stream (patrol): {exc}")
                    
                    # Event already evaluated above for overlay; use it for event notification
                    if not event and not loaded_rules:
                        # compatibility path (old check_event_match)
                        label = check_event_match(task, merged_classes)
                        event = {"label": label} if label else None

                    video_ms = get_video_time_ms(video_capture, frame_index, fps) if not use_hub else (frame_index / float(max(1, fps))) * 1000.0
                    video_ts = format_video_time_ms(video_ms)

                    if event and event.get("label"):
                        event_label = str(event["label"]).strip()
                        print(f"[worker {task_id}] 🔔 {event_label} | agent='{agent_name}' | video_time={video_ts}")
                        
                        # Handle event through session manager (with runtime video splitting)
                        if processed_frame is not None:
                            try:
                                session_manager = get_event_session_manager()
                                session_manager.handle_event_frame(
                                    agent_id=agent_id,
                                    rule_index=event.get("rule_index", 0),
                                    event_label=event_label,
                                    frame=processed_frame,
                                    camera_id=camera_id,
                                    agent_name=agent_name,
                                    detections=detections,
                                    video_timestamp=video_ts,
                                    fps=fps
                                )
                            except Exception as exc:  # noqa: BLE001
                                print(f"[worker {task_id}] ⚠️  Error handling event frame: {exc}")
                        elif loaded_rules:
                            # If processed_frame wasn't created but we have rules, create it now for event notification
                            try:
                                # Draw pose keypoints if available; otherwise draw bounding boxes
                                if detections.get("keypoints"):
                                    processed_frame = draw_pose_keypoints(frame.copy(), detections, loaded_rules)
                                else:
                                    processed_frame = draw_bounding_boxes(frame.copy(), detections, loaded_rules)
                                session_manager = get_event_session_manager()
                                session_manager.handle_event_frame(
                                    agent_id=agent_id,
                                    rule_index=event.get("rule_index", 0),
                                    event_label=event_label,
                                    frame=processed_frame,
                                    camera_id=camera_id,
                                    agent_name=agent_name,
                                    detections=detections,
                                    video_timestamp=video_ts,
                                    fps=fps
                                )
                            except Exception as exc:  # noqa: BLE001
                                print(f"[worker {task_id}] ⚠️  Error creating/handling event frame: {exc}")
                    else:
                        print(f"[worker {task_id}] ℹ️ No rule match | agent='{agent_name}' | video_time={video_ts}")

                    # Heartbeat during window
                    tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": {"updated_at": utc_now()}})
                print(f"[worker {task_id}] 💤 Patrol window ended; going back to sleep")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if not use_hub and video_capture is not None:
                video_capture.release()
        except Exception:
            pass
        print(f"[worker {task_id}] ⏹️ Exiting")


