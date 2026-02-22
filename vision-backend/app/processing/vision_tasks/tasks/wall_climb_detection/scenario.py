"""
Wall Climb Detection Scenario
-------------------------------
Person "above" wall when head/shoulders (keypoints) are above wall line. Tracker for stable IDs; VLM confirmation before alert.
Fallback: box top above line when keypoints missing. Pipeline draws red keypoints (no boxes) for confirmed.

Code layout:
  - extract_pose_frame: get pose from frame_context for VLM buffer.
  - WallClimbScenario: __init__, process (filter → track → above check → VLM defer/call → confirmed_detection_indices), get_overlay_data, reset.
"""

# -------- Imports --------
import os
import logging
from typing import Any, Dict, List, Set, Optional
from datetime import datetime

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
)
from app.processing.vision_tasks.task_lookup import register_scenario
# Use tracker WITH Kalman so same person keeps same track_id when they reappear (violator stays red)
from app.processing.vision_tasks.tasks.tracking import SimpleTracker
from app.processing.vision_tasks.tasks.class_count.counter import (
    filter_detections_by_class_with_indices,
    calculate_iou,
)
from .config import WallClimbConfig
from .wall_zone_utils import (
    is_person_above_wall_by_keypoints,
    is_box_climbing,
    check_climbing_keypoints_above_zone,
)
from .state import WallClimbDetectionState
from .types import PoseFrame, WallClimbAnalysis
from .vlm_handler import should_call_vlm, call_vlm
from app.infrastructure.external.groq_vlm_service import GroqVLMService

logger = logging.getLogger(__name__)


# ========== Helper: Extract pose frame for VLM buffer ==========

def extract_pose_frame(frame_context: ScenarioFrameContext) -> Optional[PoseFrame]:
    """
    Extract pose data from frame context for VLM buffering.
    Returns PoseFrame with frame, keypoints, person boxes, timestamp, frame_index.
    """
    detections = frame_context.detections
    keypoints_list = getattr(detections, "keypoints", None) or []
    boxes = getattr(detections, "boxes", None) or []
    classes = getattr(detections, "classes", None) or []
    
    # Filter to only person detections
    person_boxes = []
    person_keypoints = []
    
    for i, (box, cls) in enumerate(zip(boxes, classes)):
        if isinstance(cls, str) and cls.strip().lower() == "person":
            person_boxes.append(list(box))
            if i < len(keypoints_list):
                person_keypoints.append(keypoints_list[i])
            else:
                person_keypoints.append([])
    
    if not person_boxes:
        return None
    
    return PoseFrame(
        frame=frame_context.frame.copy(),
        keypoints=person_keypoints,
        person_boxes=person_boxes,
        timestamp=frame_context.timestamp,
        frame_index=frame_context.frame_index,
    )


# ========== Scenario: Wall climb (keypoints above line → VLM confirm → red keypoints) ==========

@register_scenario("wall_climb_detection")
class WallClimbScenario(BaseScenario):
    """
    Detects when a person climbs over the wall with VLM confirmation.
    
    Flow:
    1. Detect person above zone (keypoint/box based)
    2. Buffer frames for VLM
    3. Defer VLM call until we have 3 frames
    4. Call VLM with 3 frames (before, violation, after)
    5. Only emit event if VLM confirms climbing
    6. Show red keypoints (no bounding boxes) for confirmed violations
    """

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = WallClimbConfig(config, pipeline_context.task)
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
        self.state = WallClimbDetectionState(self.config_obj.buffer_size)
        self._vlm_service: Optional[GroqVLMService] = None
        os.makedirs(self.config_obj.vlm_frames_dir, exist_ok=True)
        
        # Legacy state for compatibility (will be removed after migration)
        self._state["track_side"] = {}  # track_id -> "below" | "above"
        self._state["climbed_track_ids"] = set()  # track_ids already alerted (one per person)
        self._state["red_indices"] = []  # original detection indices to draw red (deprecated - use confirmed_violations)

    def get_vlm_service(self) -> GroqVLMService:
        """Create VLM service once (lazy)."""
        if self._vlm_service is None:
            self._vlm_service = GroqVLMService()
        return self._vlm_service

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process one frame. Returns a list of events (usually 0 or 1: wall_climb_detection).
        """
        if not self.config_obj.target_class or not self.config_obj.zone_coordinates:
            self._state["red_indices"] = []
            self._state["confirmed_detection_indices"] = []
            return []

        frame = frame_context.frame
        frame_height, frame_width = frame.shape[:2]
        detections = frame_context.detections
        keypoints_list = getattr(detections, "keypoints", None) or []

        # Step 1: Extract pose frame for buffering
        pose_frame = extract_pose_frame(frame_context)
        if pose_frame:
            self.state.add_pose_frame(pose_frame)

        # Step 2: Filter detections and update tracker
        filtered_detections, original_indices = filter_detections_by_class_with_indices(
            detections, self.config_obj.target_class
        )
        valid_detections = []
        valid_indices = []
        for i, (bbox, score) in enumerate(filtered_detections):
            if score >= self.config_obj.confidence_threshold:
                valid_detections.append((bbox, score))
                valid_indices.append(original_indices[i])

        if not valid_detections:
            self._state["red_indices"] = []
            self._state["confirmed_detection_indices"] = []
            return []

        self.tracker.update(valid_detections)
        all_active_tracks = self.tracker.get_all_active_tracks()

        # Step 3: Map tracks to detection indices
        track_to_detection: Dict[int, int] = {}
        for track in all_active_tracks:
            if track.frame_id != self.tracker.frame_id:
                continue
            best_iou, best_orig_idx = 0.0, None
            for j, (det_bbox, _) in enumerate(valid_detections):
                iou = calculate_iou(track.bbox, det_bbox)
                if iou >= 0.3 and iou > best_iou:
                    best_iou = iou
                    best_orig_idx = valid_indices[j]
            if best_orig_idx is not None:
                track_to_detection[track.track_id] = best_orig_idx

        # Step 4: Process deferred VLM calls (we deferred last frame; now we have 3 frames)
        vlm_service = self.get_vlm_service()
        events: List[ScenarioEvent] = []
        to_remove = []
        
        for analysis, suspicious_frame_index in list(self.state.deferred_vlm):
            # Check if it's time to call VLM (next frame after deferral)
            if frame_context.frame_index != suspicious_frame_index + 1:
                # Still waiting for next frame
                continue
            
            # Check if we have enough frames in buffer
            if len(self.state.pose_buffer) < 3:
                print(f"[WallClimb Scenario] ⚠️ Cannot call VLM for track_id={analysis.track_id}: buffer has only {len(self.state.pose_buffer)} frames (need 3)")
                # Don't remove from deferred - will retry next frame
                continue
            
            before_frame = self.state.pose_buffer[-3].frame
            suspicious_frame = self.state.pose_buffer[-2].frame
            after_frame = self.state.pose_buffer[-1].frame
            three_frames = [before_frame, suspicious_frame, after_frame]

            print(f"[WallClimb Scenario] 📤 Calling VLM for track_id={analysis.track_id} (suspicious_frame={suspicious_frame_index}, current_frame={frame_context.frame_index}, buffer_size={len(self.state.pose_buffer)})")
            vlm_result = call_vlm(
                analysis,
                three_frames,
                self.state,
                vlm_service,
                self.config_obj.vlm_confidence_threshold,
                self.config_obj.vlm_frames_dir,
            )
            to_remove.append((analysis, suspicious_frame_index))

            if vlm_result:
                print(f"[WallClimb Scenario] 📥 VLM response: climbing_detected={vlm_result.climbing_detected} confidence={vlm_result.confidence:.2f} description={vlm_result.description[:80]!r}...")
            else:
                print(f"[WallClimb Scenario] 📥 VLM response: no confirmation (None or not climbing)")

            if vlm_result and vlm_result.climbing_detected:
                # VLM confirmed - add to confirmed violations and emit event
                self.state.confirmed_violations.add(vlm_result.track_id)
                self.state.confirmed_track_to_detection[vlm_result.track_id] = vlm_result.person_index
                
                # Check cooldown before emitting event
                last_emitted = self._state.get("last_emitted_time", {}).get(vlm_result.track_id)
                if last_emitted is None or (frame_context.timestamp - last_emitted).total_seconds() > self.config_obj.alert_cooldown_seconds:
                    label = self.generate_label(1)
                    print(f"[WallClimb Scenario] 🚨 EVENT: {label} emitted (track_id={vlm_result.track_id} confidence={vlm_result.confidence:.2f})")
                    events.append(
                        ScenarioEvent(
                            event_type="wall_climb_detection",
                            label=label,
                            confidence=vlm_result.confidence,
                            metadata={
                                "target_class": self.config_obj.target_class,
                                "track_id": vlm_result.track_id,
                                "vlm_description": vlm_result.description,
                                "vlm_response": vlm_result.vlm_response,
                            },
                            detection_indices=[vlm_result.person_index],
                            timestamp=vlm_result.timestamp,
                            frame_index=vlm_result.frame_index,
                        )
                    )
                    if "last_emitted_time" not in self._state:
                        self._state["last_emitted_time"] = {}
                    self._state["last_emitted_time"][vlm_result.track_id] = frame_context.timestamp

        for item in to_remove:
            self.state.deferred_vlm.remove(item)

        # Step 5: Detect persons above zone and create analyses
        track_side: Dict[int, str] = self._state.get("track_side") or {}
        current_analyses: List[WallClimbAnalysis] = []
        
        for track in all_active_tracks:
            track_id = track.track_id
            orig_idx = track_to_detection.get(track_id)
            if orig_idx is None:
                continue

            # Check if climbing keypoints (head, shoulders, wrists) are above zone
            is_above = False
            detected_keypoint_indices = []
            
            if orig_idx < len(keypoints_list) and keypoints_list[orig_idx]:
                # Check specific keypoints: head (0), shoulders (5,6), wrists (9,10)
                is_above, detected_keypoint_indices = check_climbing_keypoints_above_zone(
                    keypoints_list[orig_idx],
                    self.config_obj.zone_coordinates,
                    frame_width,
                    frame_height,
                    kp_confidence_threshold=0.25,
                )
                
                # Fallback: if no keypoints detected above, check with legacy method
                if not is_above:
                    is_above = is_person_above_wall_by_keypoints(
                        keypoints_list[orig_idx],
                        self.config_obj.zone_coordinates,
                        frame_width,
                        frame_height,
                        kp_confidence_threshold=0.25,
                    )
            
            # Final fallback: use bounding box if keypoints not available
            if not is_above:
                for j, oidx in enumerate(valid_indices):
                    if oidx == orig_idx and j < len(valid_detections):
                        is_above = is_box_climbing(
                            valid_detections[j][0],
                            self.config_obj.zone_coordinates,
                            frame_width,
                            frame_height,
                        )
                        break

            current_side = "above" if is_above else "below"
            prev_side = track_side.get(track_id)
            track_side[track_id] = current_side

            # If climbing keypoints detected above zone, create analysis and trigger VLM
            if is_above:
                # Get bounding box
                bbox = None
                for j, oidx in enumerate(valid_indices):
                    if oidx == orig_idx and j < len(valid_detections):
                        bbox = valid_detections[j][0]
                        break
                
                if bbox:
                    analysis = WallClimbAnalysis(
                        track_id=track_id,
                        person_index=orig_idx,
                        box=list(bbox),
                        is_above_zone=True,
                        confidence=1.0,
                        timestamp=frame_context.timestamp,
                        frame_index=frame_context.frame_index,
                    )
                    current_analyses.append(analysis)
                    
                    # Check if we need to call VLM
                    # Case 1: New violation (below -> above transition) - always try to call VLM
                    # Case 2: Already above but not VLM-confirmed yet - call VLM if not throttled
                    # Case 3: Already above and VLM-confirmed - skip (already confirmed)
                    
                    is_vlm_confirmed = track_id in self.state.confirmed_violations
                    is_new_violation = prev_side == "below" and current_side == "above"
                    is_already_above = prev_side == "above" and current_side == "above"
                    is_first_detection = prev_side is None  # Person just appeared above zone
                    
                    # Log detected keypoints for debugging
                    if detected_keypoint_indices:
                        kp_names = {0: "head", 5: "left_shoulder", 6: "right_shoulder", 9: "left_wrist", 10: "right_wrist"}
                        detected_names = [kp_names.get(idx, f"kp_{idx}") for idx in detected_keypoint_indices]
                        print(f"[WallClimb Scenario] 🔍 track_id={track_id} detected keypoints above zone: {detected_names} (indices: {detected_keypoint_indices})")
                    
                    # Check if VLM should be called
                    # Strategy: Trigger VLM whenever keypoints are detected above zone (continuous monitoring)
                    # Skip only if: already VLM-confirmed OR throttled/cached
                    should_call = False
                    reason = ""
                    
                    if is_vlm_confirmed:
                        # Already confirmed: skip VLM call (no logging needed, this is normal)
                        pass
                    else:
                        # Not confirmed yet: try to call VLM whenever keypoints are detected above
                        # This ensures continuous monitoring and VLM triggering
                        if should_call_vlm(analysis, self.state, self.config_obj.vlm_throttle_seconds):
                            should_call = True
                            if is_new_violation:
                                reason = f"new violation - keypoints above: {detected_keypoint_indices}"
                            elif is_first_detection:
                                reason = f"first detection - keypoints above: {detected_keypoint_indices}"
                            elif is_already_above:
                                reason = f"continuous monitoring - keypoints above: {detected_keypoint_indices}"
                            else:
                                reason = f"keypoints detected above zone: {detected_keypoint_indices}"
                        else:
                            print(f"[WallClimb Scenario] ⏭️ Skipping VLM for track_id={track_id}: throttled/cached (keypoints above: {detected_keypoint_indices})")
                    
                    if should_call:
                        # Check if already deferred (avoid duplicates)
                        already_deferred = any(
                            a.track_id == track_id and f_idx == frame_context.frame_index
                            for a, f_idx in self.state.deferred_vlm
                        )
                        if not already_deferred:
                            # Check if we have enough frames in buffer
                            if len(self.state.pose_buffer) >= 2:  # Need at least 2 frames to defer (will have 3 on next frame)
                                # Defer VLM call until next frame (so we have 3 frames)
                                self.state.deferred_vlm.append((analysis, frame_context.frame_index))
                                print(f"[WallClimb Scenario] 📋 Deferred VLM for track_id={track_id} (reason: {reason}, will call on next frame, buffer_size={len(self.state.pose_buffer)})")
                            else:
                                print(f"[WallClimb Scenario] ⚠️ Cannot defer VLM for track_id={track_id}: buffer has only {len(self.state.pose_buffer)} frames (need at least 2)")
                        else:
                            print(f"[WallClimb Scenario] ⏭️ VLM already deferred for track_id={track_id} at frame {frame_context.frame_index}")
            
            # Do NOT clear confirmed_violations when person moves below zone.
            # Once VLM confirms violation, that track_id stays red until track is lost (so only the violator is highlighted).

        # Update state
        active_ids = {t.track_id for t in all_active_tracks}
        self._state["track_side"] = {tid: s for tid, s in track_side.items() if tid in active_ids}

        # Remove confirmed_violations only when track is no longer in scene (so red doesn't stick to dead IDs)
        self.state.confirmed_violations &= active_ids
        for tid in list(self.state.confirmed_track_to_detection.keys()):
            if tid not in active_ids:
                self.state.confirmed_track_to_detection.pop(tid, None)

        # Build confirmed_detection_indices from CURRENT frame: for each confirmed track_id, use this frame's detection index.
        # This ensures only the violating person(s) are red, and correct person when 2–3 people are present.
        confirmed_detection_indices = []
        for track_id in self.state.confirmed_violations:
            current_orig_idx = track_to_detection.get(track_id)
            if current_orig_idx is not None:
                confirmed_detection_indices.append(current_orig_idx)
                self.state.confirmed_track_to_detection[track_id] = current_orig_idx
        self._state["confirmed_detection_indices"] = confirmed_detection_indices
        
        # Legacy red_indices for backward compatibility (will be empty - no bounding boxes)
        self._state["red_indices"] = []

        # Step 6: Clean up old data
        self.state.cleanup_old_data(frame_context.timestamp)

        return events

    def generate_label(self, count: int) -> str:
        if self.config_obj.custom_label and isinstance(self.config_obj.custom_label, str) and self.config_obj.custom_label.strip():
            return self.config_obj.custom_label.strip()
        if count == 1:
            return "Person climbed over wall"
        return f"{count} persons climbed over wall"

    def reset(self) -> None:
        """Clear all state when rule is disabled."""
        self.state.reset()
        self._vlm_service = None
        self._state["track_side"] = {}
        self._state["climbed_track_ids"] = set()
        self._state["red_indices"] = []
        self._state["confirmed_detection_indices"] = []
        self._state["last_emitted_time"] = {}

    def requires_yolo_detections(self) -> bool:
        """We need YOLO pose (person + keypoints)."""
        return True
