"""
Fall detection scenario
------------------------

Real-time fall detection using a state machine and FPS-independent metrics with VLM confirmation:

  NORMAL → FALL_SUSPECTED → [VLM Analysis] → (RECOVERED → NORMAL) or (VLM_CONFIRMED → CONFIRMED_FALL → ALERT)

- Fall suspected: sudden downward movement (A) + (height collapse (B) or horizontal torso (C)).
- VLM confirmation: sends 5 frames to VLM for confirmation (API limit).
- Recovery: height increased, torso vertical, or hip/head moved up.
- Confirmed fall: VLM confirms fall → emit alert.

Keypoints: FALL_SUSPECTED → orange, CONFIRMED_FALL → red (set in state for pipeline/UI).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioEvent,
    ScenarioFrameContext,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from app.processing.vision_tasks.tracking import SimpleTracker
from .config import FallDetectionConfig
from .types import PoseFrame, FallAnalysis
from .state import FallDetectionState
from .vlm_handler import should_call_vlm, call_vlm
from app.infrastructure.external.groq_vlm_service import GroqVLMService

# -----------------------------------------------------------------------------
# State labels (for readability)
# -----------------------------------------------------------------------------
STATE_NORMAL = "normal"
STATE_FALL_SUSPECTED = "fall_suspected"
STATE_CONFIRMED_FALL = "confirmed_fall"

# -----------------------------------------------------------------------------
# Keypoint helpers (COCO 17: 0=nose, 5/6=shoulders, 11/12=hips, etc.)
# -----------------------------------------------------------------------------


def get_keypoint(
    person_keypoints: List[List[float]],
    idx: int,
    confidence_threshold: float = 0.25,
) -> Optional[Tuple[float, float]]:
    if idx >= len(person_keypoints):
        return None
    kp = person_keypoints[idx]
    if not kp or len(kp) < 2:
        return None
    if len(kp) >= 3 and kp[2] < confidence_threshold:
        return None
    return float(kp[0]), float(kp[1])


def midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)


def angle_from_vertical(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Angle (degrees) of segment p1->p2 from vertical (0 = vertical, 90 = horizontal)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return 0.0
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)


def keypoint_bbox(person_keypoints: List[List[float]]) -> Tuple[float, float, float, float]:
    """Return (min_x, min_y, width, height) from all valid keypoints."""
    xs, ys = [], []
    for kp in person_keypoints:
        if kp and len(kp) >= 2:
            if len(kp) < 3 or (kp[2] >= 0.2):
                xs.append(float(kp[0]))
                ys.append(float(kp[1]))
    if not xs or not ys:
        return 0.0, 0.0, 0.0, 0.0
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return min_x, min_y, max_x - min_x, max_y - min_y


def get_head_point(
    person_keypoints: List[List[float]],
    confidence_threshold: float,
) -> Optional[Tuple[float, float]]:
    """Head position: nose (0) or midpoint of eyes (1,2)."""
    nose = get_keypoint(person_keypoints, 0, confidence_threshold)
    if nose:
        return nose
    le = get_keypoint(person_keypoints, 1, confidence_threshold)
    re = get_keypoint(person_keypoints, 2, confidence_threshold)
    if le and re:
        return midpoint(le, re)
    return None


def analyze_person_metrics(
    person_keypoints: List[List[float]],
    prev_metrics: Optional[Dict[str, Any]] = None,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    config: Optional[FallDetectionConfig] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, bool]]:
    """
    Compute current pose metrics and multiple fall detection signals.

    Returns:
        metrics: dict with hip_y, head_y, height, angle, aspect_ratio, etc.
        signals: dict with fall detection signals (hip_drop, head_drop, height_collapse, 
                torso_horizontal, aspect_ratio_flipped, etc.)
    """
    if config is None:
        return None, {}
    conf = config.kp_confidence_threshold
    left_shoulder = get_keypoint(person_keypoints, 5, conf)
    right_shoulder = get_keypoint(person_keypoints, 6, conf)
    left_hip = get_keypoint(person_keypoints, 11, conf)
    right_hip = get_keypoint(person_keypoints, 12, conf)
    if not (left_shoulder and right_shoulder and left_hip and right_hip):
        return None, {}

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    head_pt = get_head_point(person_keypoints, conf)
    min_x, min_y, kp_width, kp_height = keypoint_bbox(person_keypoints)
    if kp_height <= 0:
        kp_height = 1.0
    if kp_width <= 0:
        kp_width = 1.0
    
    angle = angle_from_vertical(shoulder_mid, hip_mid)
    aspect_ratio = kp_width / kp_height if kp_height > 0 else 0.0
    
    # Head-hip vertical separation ratio (when lying, head and hip are at similar level)
    head_hip_vertical_ratio = 1.0
    if head_pt and kp_height > 0:
        head_hip_sep = abs(head_pt[1] - hip_mid[1])
        head_hip_vertical_ratio = head_hip_sep / kp_height

    metrics = {
        "hip_y": hip_mid[1],
        "head_y": head_pt[1] if head_pt else hip_mid[1],
        "height": kp_height,
        "width": kp_width,
        "angle": angle,
        "aspect_ratio": aspect_ratio,
        "head_hip_vertical_ratio": head_hip_vertical_ratio,
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
        "head_pt": head_pt,
    }

    signals = {
        "hip_drop_absolute": False,
        "hip_drop_relative": False,
        "head_drop_absolute": False,
        "head_drop_relative": False,
        "height_collapse": False,
        "torso_horizontal": False,
        "aspect_ratio_flipped": False,
        "head_hip_aligned": False,
        "velocity_drop": False,
    }

    if prev_metrics is not None:
        prev_height = prev_metrics.get("height") or kp_height
        prev_hip_y = prev_metrics.get("hip_y")
        prev_head_y = prev_metrics.get("head_y")
        dt_sec = prev_metrics.get("dt_sec")

        # 1. Absolute pixel drop (works at any distance)
        if prev_hip_y is not None:
            hip_drop_pixels = hip_mid[1] - prev_hip_y  # Positive = downward
            if hip_drop_pixels > config.hip_drop_pixels_threshold:
                signals["hip_drop_absolute"] = True
            
            # Relative drop (normalized by height)
            if prev_height > 0:
                hip_drop_ratio = hip_drop_pixels / prev_height
                if hip_drop_ratio > config.hip_drop_ratio_threshold:
                    signals["hip_drop_relative"] = True

        if prev_head_y is not None and head_pt:
            head_drop_pixels = head_pt[1] - prev_head_y  # Positive = downward
            if head_drop_pixels > config.head_drop_pixels_threshold:
                signals["head_drop_absolute"] = True
            
            # Relative drop
            if prev_height > 0:
                head_drop_ratio = head_drop_pixels / prev_height
                if head_drop_ratio > config.head_drop_ratio_threshold:
                    signals["head_drop_relative"] = True

        # 2. Height collapse
        if prev_height > 0:
            drop_ratio = (prev_height - kp_height) / prev_height
            if drop_ratio > config.height_collapse_ratio:
                signals["height_collapse"] = True

        # 3. Velocity-based drop (FPS-independent)
        if dt_sec is not None and dt_sec > 0 and prev_height > 0 and prev_hip_y is not None:
            velocity = (hip_mid[1] - prev_hip_y) / prev_height / dt_sec
            if velocity > config.sudden_drop_velocity_threshold:
                signals["velocity_drop"] = True

    # 4. Torso horizontal (current frame only, no history needed)
    if angle > config.torso_horizontal_angle_threshold:
        signals["torso_horizontal"] = True

    # 5. Aspect ratio flipped (person wider than tall = horizontal)
    if aspect_ratio > config.aspect_ratio_threshold:
        signals["aspect_ratio_flipped"] = True

    # 6. Head-hip vertical alignment (when lying, head and hip are at similar level)
    if head_hip_vertical_ratio < config.head_hip_vertical_ratio_max:
        signals["head_hip_aligned"] = True

    return metrics, signals


def check_recovery(
    metrics: Dict[str, Any],
    fall_height: float,
    fall_hip_y: float,
    fall_head_y: Optional[float],
    config: FallDetectionConfig,
) -> bool:
    """
    True if person has recovered (standing up again).
    
    Recovery means:
    1. Height increased significantly (person standing)
    2. Torso is vertical (not horizontal)
    3. Hip/head moved up significantly from fall position
    """
    if fall_height <= 0:
        return True
    
    current_height = metrics.get("height") or 0
    current_hip_y = metrics.get("hip_y")
    current_head_y = metrics.get("head_y")
    current_angle = metrics.get("angle", 90)
    current_aspect = metrics.get("aspect_ratio") or 0.0  # < 1 when standing (taller than wide)
    torso_vertical = current_angle <= config.recovery_torso_angle_max
    angle_not_horizontal = current_angle <= 50.0
    
    # 0. Clearly standing: tall + vertical torso → recover (red keypoints go back to normal)
    min_standing = getattr(config, "min_height_for_standing", 140)
    if current_height >= min_standing and torso_vertical:
        return True
    # 0b. Upright bbox (aspect < 1) + not horizontal angle → standing
    if current_aspect < 1.0 and angle_not_horizontal and current_height >= 100:
        return True
    
    # 1. Height recovery: person is standing again (current much taller than fall height)
    if current_height >= fall_height * config.recovery_height_ratio:
        return True
    
    # 2. Torso vertical: person is upright (but might still be on ground)
    
    # 3. Hip/head lifted up: moved up significantly from fall position
    hip_lifted = False
    head_lifted = False
    
    if current_hip_y is not None:
        hip_lift_pixels = fall_hip_y - current_hip_y  # Positive = moved up
        if fall_height > 0:
            hip_lift_ratio = hip_lift_pixels / fall_height
            if hip_lift_ratio >= config.recovery_hip_lift_ratio:
                hip_lifted = True
    
    if current_head_y is not None and fall_head_y is not None:
        head_lift_pixels = fall_head_y - current_head_y  # Positive = moved up
        if fall_height > 0:
            head_lift_ratio = head_lift_pixels / fall_height
            if head_lift_ratio >= config.recovery_hip_lift_ratio:
                head_lifted = True
    
    # Recovery = torso vertical AND (hip lifted OR head lifted)
    if torso_vertical and (hip_lifted or head_lifted):
        return True
    
    # Also recover if height is good AND torso is vertical (strong signal)
    if current_height >= fall_height * 1.15 and torso_vertical:
        return True
    
    return False


# -----------------------------------------------------------------------------
# Scenario
# -----------------------------------------------------------------------------


def filter_detections_by_class_with_indices(
    detections, target_class: str
) -> Tuple[List[Tuple[List[float], float]], List[int]]:
    """
    Filter detections by class while preserving original detection indices.
    
    Returns:
        (filtered_detections, original_indices) where:
        - filtered_detections: List of (bbox, score) tuples for tracker
        - original_indices: List of original detection indices matching filtered_detections
    """
    boxes = detections.boxes
    classes = detections.classes
    scores = detections.scores
    filtered: List[Tuple[List[float], float]] = []
    original_indices: List[int] = []
    class_name_lower = target_class.lower()
    
    for i, detected_class in enumerate(classes):
        if isinstance(detected_class, str) and detected_class.lower() == class_name_lower:
            if i < len(boxes) and i < len(scores):
                filtered.append((list(boxes[i]), float(scores[i])))
                original_indices.append(i)
    
    return filtered, original_indices


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Find intersection
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


@register_scenario("fall_detection")
class FallDetectionScenario(BaseScenario):
    """
    Detects human falls using a state machine and FPS-independent pose analysis with VLM confirmation.

    - FALL SUSPECTED: sudden drop + (height collapse or horizontal torso).
    - VLM confirmation: sends 9 frames (4 before + suspected + 4 after) to VLM.
    - CONFIRMED FALL: VLM confirms fall → alert; keypoints orange (suspected), red (confirmed).
    """

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = FallDetectionConfig(config, pipeline_context.task)
        
        # Initialize tracker for stable person tracking (like class_count/box_count)
        self.tracker = SimpleTracker(
            max_age=30,  # Keep tracks for 30 frames after last detection
            min_hits=3,  # Require 3 hits before confirming track
            iou_threshold=0.3,  # IoU threshold for matching
            score_threshold=self.config_obj.confidence_threshold
        )
        
        # VLM state management
        if self.config_obj.vlm_enabled:
            self.vlm_state = FallDetectionState(self.config_obj.vlm_buffer_size)
            self._vlm_service: Optional[GroqVLMService] = None
            os.makedirs(self.config_obj.vlm_frames_dir, exist_ok=True)
        else:
            self.vlm_state = None
            self._vlm_service = None
        
        # State storage by track_id (stable across frames) instead of detection index
        self._state["history"] = {}  # track_id -> last frame metrics (with dt_sec)
        self._state["baseline"] = {}  # track_id -> baseline metrics (standing height, hip_y, etc.)
        self._state["person_states"] = {}  # track_id -> STATE_*
        self._state["fall_suspected_at"] = {}  # track_id -> datetime
        self._state["fall_height"] = {}  # track_id -> float
        self._state["fall_hip_y"] = {}  # track_id -> float (hip Y when fall suspected)
        self._state["fall_head_y"] = {}  # track_id -> float (head Y when fall suspected)
        self._state["last_frame_time"] = None  # for dt
        self._state["last_alert_time"] = None
        self._state["fall_suspected_indices"] = []  # detection indices for drawing (orange)
        self._state["fall_confirmed_indices"] = []  # detection indices for drawing (red)

    def get_vlm_service(self) -> Optional[GroqVLMService]:
        """Create VLM service once (lazy)."""
        if not self.config_obj.vlm_enabled:
            return None
        if self._vlm_service is None:
            self._vlm_service = GroqVLMService()
        return self._vlm_service

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        if not self.config_obj.enabled or not self.config_obj.target_class:
            self._state["fall_suspected_indices"] = []
            self._state["fall_confirmed_indices"] = []
            return []

        # Step 1: Extract pose frame for VLM buffering (if VLM enabled)
        if self.config_obj.vlm_enabled and self.vlm_state:
            pose_frame = extract_pose_frame(frame_context)
            if pose_frame:
                self.vlm_state.add_pose_frame(pose_frame)

        detections = frame_context.detections
        target_class = self.config_obj.target_class
        now = frame_context.timestamp
        keypoints = detections.keypoints or []

        # Filter detections by class while preserving original indices
        filtered_detections, original_indices = filter_detections_by_class_with_indices(
            detections, target_class
        )
        
        # Filter by confidence threshold
        valid_indices = []
        valid_detections = []
        for i, (bbox, score) in enumerate(filtered_detections):
            if score >= self.config_obj.confidence_threshold:
                valid_indices.append(original_indices[i])
                valid_detections.append((bbox, score))

        if not valid_detections:
            # Clear state for persons no longer detected (but keep baseline for a bit)
            self._state["history"].clear()
            self._state["person_states"].clear()
            self._state["fall_suspected_at"].clear()
            self._state["fall_height"].clear()
            self._state["fall_hip_y"].clear()
            self._state["fall_head_y"].clear()
            self._state["fall_suspected_indices"] = []
            self._state["fall_confirmed_indices"] = []
            self._state["last_frame_time"] = None
            return []

        # Update tracker with filtered detections
        active_tracks = self.tracker.update(valid_detections)
        all_active_tracks = self.tracker.get_all_active_tracks()

        # Map tracks to detection indices using IoU matching
        # track_id -> detection_index mapping for this frame
        track_to_detection_map: Dict[int, int] = {}
        for track in all_active_tracks:
            # Track was updated this frame (tracker.frame_id is incremented after update)
            if track.frame_id == self.tracker.frame_id:
                track_bbox = track.bbox
                best_iou = 0.0
                best_idx = None
                for i, (det_bbox, _) in enumerate(valid_detections):
                    iou = calculate_iou(track_bbox, det_bbox)
                    if iou > best_iou and iou >= 0.3:  # Match threshold
                        best_iou = iou
                        best_idx = valid_indices[i]  # Original detection index
                if best_idx is not None:
                    track_to_detection_map[track.track_id] = best_idx

        # dt for FPS-independent velocity (seconds)
        dt_sec = None
        if self._state.get("last_frame_time") is not None:
            prev_time = self._state["last_frame_time"]
            if isinstance(prev_time, datetime) and isinstance(now, datetime):
                dt_sec = (now - prev_time).total_seconds()
            if dt_sec is not None and (dt_sec <= 0 or dt_sec > 2.0):
                dt_sec = None  # skip bad or huge gaps
        self._state["last_frame_time"] = now

        history = self._state["history"]
        person_states = self._state["person_states"]
        fall_suspected_at = self._state["fall_suspected_at"]
        fall_height = self._state["fall_height"]
        cooldown_sec = self.config_obj.alert_cooldown_seconds

        suspected_detection_indices = []  # Detection indices for drawing (orange)
        confirmed_detection_indices = []  # Detection indices for drawing (red)
        events = []

        # Step 2: Process deferred VLM calls (if VLM enabled)
        # We deferred VLM calls when fall was suspected; now check if we have enough frames
        if self.config_obj.vlm_enabled and self.vlm_state:
            vlm_service = self.get_vlm_service()
            if vlm_service:
                to_remove = []
                buf_len_now = len(self.vlm_state.pose_buffer)
                for analysis, buffer_len_at_suspect in list(self.vlm_state.deferred_vlm):
                    # Buffer-relative: need 2+ new frames after suspect and at least 5 frames total (Groq API max 5 images)
                    if buf_len_now < 5 or buf_len_now < buffer_len_at_suspect + 2:
                        continue
                    # Take last 5 frames from buffer for VLM
                    recent = self.vlm_state.pose_buffer[-5:]
                    frames_to_send = [pf.frame for pf in recent]
                    if len(frames_to_send) < 5:
                        continue
                    print(f"[Fall Detection] 📤 Calling VLM for track_id={analysis.track_id} (buffer_len={buf_len_now}, frames_collected={len(frames_to_send)})")
                    vlm_result = call_vlm(
                        analysis,
                        frames_to_send,
                        self.vlm_state,
                        vlm_service,
                        self.config_obj.vlm_confidence_threshold,
                        self.config_obj.vlm_frames_dir,
                    )
                    to_remove.append((analysis, buffer_len_at_suspect))
                    
                    if vlm_result and vlm_result.fall_detected:
                        # VLM confirmed fall - transition to CONFIRMED_FALL
                        # Keep fall_height, fall_hip_y, fall_head_y so recovery can detect when person stands
                        track_id = analysis.track_id
                        person_states[track_id] = STATE_CONFIRMED_FALL
                        self.vlm_state.confirmed_falls.add(track_id)
                        self.vlm_state.confirmed_track_to_detection[track_id] = analysis.person_index
                        fall_suspected_at.pop(track_id, None)
                        print(f"[Fall Detection] 🔴 VLM CONFIRMED FALL (Track {track_id}): confidence={vlm_result.confidence:.2f}")
                    elif vlm_result:
                        # VLM said no fall - return to NORMAL (false positive)
                        track_id = analysis.track_id
                        if person_states.get(track_id) == STATE_FALL_SUSPECTED:
                            person_states[track_id] = STATE_NORMAL
                            fall_suspected_at.pop(track_id, None)
                            fall_height.pop(track_id, None)
                            self._state["fall_hip_y"].pop(track_id, None)
                            self._state["fall_head_y"].pop(track_id, None)
                            print(f"[Fall Detection] ✅ VLM cleared false positive (Track {track_id}): confidence={vlm_result.confidence:.2f}")
                
                for item in to_remove:
                    try:
                        self.vlm_state.deferred_vlm.remove(item)
                    except ValueError:
                        pass
                
                # Cleanup old data
                self.vlm_state.cleanup_old_data(now)

        # Process each tracked person using track_id (stable across frames)
        for track in all_active_tracks:
            track_id = track.track_id
            
            # Get detection index for this track (for keypoints access)
            detection_idx = track_to_detection_map.get(track_id)
            if detection_idx is None or detection_idx >= len(keypoints):
                continue  # Skip if no matching detection or invalid index
            
            person_kps = keypoints[detection_idx]
            if not person_kps:
                continue

            # Get state and metrics for this track_id
            prev_metrics = history.get(track_id)
            baseline_metrics = self._state["baseline"].get(track_id)
            
            if prev_metrics is not None and dt_sec is not None:
                prev_metrics = {**prev_metrics, "dt_sec": dt_sec}

            metrics, signals = analyze_person_metrics(
                person_kps, prev_metrics, baseline_metrics, self.config_obj
            )
            if metrics is None:
                continue

            # Update baseline if person is in normal state (standing)
            state = person_states.get(track_id, STATE_NORMAL)
            if state == STATE_NORMAL:
                # Update baseline with current metrics (smooth baseline for stability)
                if baseline_metrics is None:
                    self._state["baseline"][track_id] = {}
                baseline = self._state["baseline"][track_id]
                # Store baseline height (average over time for stability)
                if "height" not in baseline:
                    baseline["height"] = metrics.get("height", 0)
                else:
                    # Smooth baseline (exponential moving average)
                    baseline["height"] = 0.7 * baseline["height"] + 0.3 * metrics.get("height", baseline["height"])

            # Store metrics for next frame (with dt_sec for next iteration)
            history[track_id] = {**metrics, "dt_sec": None}
            state = person_states.get(track_id, STATE_NORMAL)

            # ----- State transitions -----
            if state == STATE_NORMAL:
                # Person is in normal state - ensure they're NOT in confirmed/suspected lists
                # (defensive check to prevent stale state from previous frames)
                if detection_idx in confirmed_detection_indices:
                    confirmed_detection_indices.remove(detection_idx)
                if detection_idx in suspected_detection_indices:
                    suspected_detection_indices.remove(detection_idx)
                
                # Trigger FALL_SUSPECTED using flexible logic:
                # Strong signal: (hip drop OR head drop) AND (torso horizontal OR aspect ratio flipped)
                # OR multiple moderate signals
                
                strong_drop = (
                    signals.get("hip_drop_absolute") or 
                    signals.get("hip_drop_relative") or 
                    signals.get("head_drop_absolute") or 
                    signals.get("head_drop_relative") or
                    signals.get("velocity_drop")
                )
                
                horizontal_posture = (
                    signals.get("torso_horizontal") or 
                    signals.get("aspect_ratio_flipped") or
                    signals.get("head_hip_aligned")
                )
                
                height_signal = signals.get("height_collapse")
                
                # Trigger if: (strong drop + horizontal) OR (drop + height collapse + horizontal)
                # OR: person is clearly horizontal (lying down) even without drop signal (catches missed transitions)
                fall_detected = False
                if strong_drop and horizontal_posture:
                    fall_detected = True
                    print(f"[fall_detection] 🟠 FALL SUSPECTED (Track {track_id}): Strong drop + horizontal posture")
                elif strong_drop and height_signal:
                    fall_detected = True
                    print(f"[fall_detection] 🟠 FALL SUSPECTED (Track {track_id}): Strong drop + height collapse")
                elif (signals.get("hip_drop_relative") or signals.get("head_drop_relative")) and height_signal and horizontal_posture:
                    fall_detected = True
                    print(f"[fall_detection] 🟠 FALL SUSPECTED (Track {track_id}): Relative drop + height collapse + horizontal")
                elif prev_metrics is None and horizontal_posture and metrics.get("height", 0) < 150:
                    # Fallback: person is horizontal and low height (likely lying down) even without drop history
                    # This catches cases where we missed the fall transition due to frame skipping
                    fall_detected = True
                    print(f"[fall_detection] 🟠 FALL SUSPECTED (Track {track_id}): Horizontal posture + low height (no drop history)")
                elif horizontal_posture and metrics.get("height", 0) < 280:
                    # Already lying (padukura): clearly horizontal + low height → trigger so full fall alerts work
                    fall_detected = True
                    print(f"[fall_detection] 🟠 FALL SUSPECTED (Track {track_id}): Already lying (horizontal + low height)")
                
                if fall_detected:
                    person_states[track_id] = STATE_FALL_SUSPECTED
                    fall_suspected_at[track_id] = now
                    fall_height[track_id] = metrics["height"]
                    self._state["fall_hip_y"][track_id] = metrics["hip_y"]
                    self._state["fall_head_y"][track_id] = metrics.get("head_y")
                    suspected_detection_indices.append(detection_idx)
                    
                    # If VLM enabled, create analysis and defer VLM call
                    if self.config_obj.vlm_enabled and self.vlm_state:
                        # Get bounding box
                        bbox = None
                        for j, oidx in enumerate(valid_indices):
                            if oidx == detection_idx and j < len(valid_detections):
                                bbox = valid_detections[j][0]
                                break
                        
                        if bbox:
                            analysis = FallAnalysis(
                                track_id=track_id,
                                person_index=detection_idx,
                                box=list(bbox),
                                metrics=metrics,
                                signals=signals,
                                confidence=1.0,
                                timestamp=now,
                                frame_index=frame_context.frame_index,
                            )
                            
                            # Check if we should call VLM (throttling)
                            if should_call_vlm(analysis, self.vlm_state, self.config_obj.vlm_throttle_seconds):
                                # Defer by buffer length: call VLM once we have 2+ more frames in buffer (works when frames are skipped)
                                buffer_len_now = len(self.vlm_state.pose_buffer)
                                already_deferred = any(
                                    a.track_id == track_id for a, _ in self.vlm_state.deferred_vlm
                                )
                                if not already_deferred:
                                    self.vlm_state.deferred_vlm.append((analysis, buffer_len_now))
                                    print(f"[Fall Detection] 📋 Deferred VLM for track_id={track_id} (buffer_len={buffer_len_now}, will call when buffer has 2+ more frames)")
                            else:
                                print(f"[Fall Detection] ⏭️ Skipping VLM for track_id={track_id}: throttled/cached")

            elif state == STATE_FALL_SUSPECTED:
                suspected_detection_indices.append(detection_idx)
                stored_fall_height = fall_height.get(track_id) or metrics["height"]
                stored_fall_hip_y = self._state["fall_hip_y"].get(track_id, metrics["hip_y"])
                stored_fall_head_y = self._state["fall_head_y"].get(track_id)

                # Check for recovery (person getting up)
                if check_recovery(metrics, stored_fall_height, stored_fall_hip_y, stored_fall_head_y, self.config_obj):
                    # Recovered → back to NORMAL
                    person_states[track_id] = STATE_NORMAL
                    fall_suspected_at.pop(track_id, None)
                    fall_height.pop(track_id, None)
                    self._state["fall_hip_y"].pop(track_id, None)
                    self._state["fall_head_y"].pop(track_id, None)
                    # Remove from VLM deferred list if present
                    if self.config_obj.vlm_enabled and self.vlm_state:
                        self.vlm_state.deferred_vlm = [
                            (a, buf_len) for a, buf_len in self.vlm_state.deferred_vlm
                            if a.track_id != track_id
                        ]
                    if detection_idx in suspected_detection_indices:
                        suspected_detection_indices.remove(detection_idx)
                # If VLM enabled, wait for VLM confirmation (no timeout)
                # If VLM disabled, use timeout-based confirmation (legacy mode)
                elif not self.config_obj.vlm_enabled:
                    # Legacy mode: use timeout
                    fall_time = fall_suspected_at.get(track_id)
                    if fall_time is not None and isinstance(now, datetime) and isinstance(fall_time, datetime):
                        elapsed = (now - fall_time).total_seconds()
                        recovery_timeout = self.config_obj.recovery_timeout_seconds
                        if elapsed >= recovery_timeout:
                            # Timeout → CONFIRMED_FALL (legacy mode only)
                            person_states[track_id] = STATE_CONFIRMED_FALL
                            fall_suspected_at.pop(track_id, None)
                            fall_height.pop(track_id, None)
                            self._state["fall_hip_y"].pop(track_id, None)
                            self._state["fall_head_y"].pop(track_id, None)
                            if detection_idx in suspected_detection_indices:
                                suspected_detection_indices.remove(detection_idx)
                            confirmed_detection_indices.append(detection_idx)
                            print(f"[fall_detection] 🔴 CONFIRMED FALL (Track {track_id}): No recovery after {elapsed:.1f}s (legacy timeout mode)")
                # If VLM enabled, wait for VLM confirmation (handled in deferred VLM processing above)

            elif state == STATE_CONFIRMED_FALL:
                # Person is confirmed fallen - check for recovery
                stored_fall_height = fall_height.get(track_id) or metrics["height"]
                stored_fall_hip_y = self._state["fall_hip_y"].get(track_id, metrics["hip_y"])
                stored_fall_head_y = self._state["fall_head_y"].get(track_id)
                
                if check_recovery(metrics, stored_fall_height, stored_fall_hip_y, stored_fall_head_y, self.config_obj):
                    # Person stood back up - clear confirmed state so keypoints return to normal (remove red)
                    person_states[track_id] = STATE_NORMAL
                    fall_height.pop(track_id, None)
                    self._state["fall_hip_y"].pop(track_id, None)
                    self._state["fall_head_y"].pop(track_id, None)
                    if self.config_obj.vlm_enabled and self.vlm_state:
                        self.vlm_state.confirmed_falls.discard(track_id)
                        self.vlm_state.confirmed_track_to_detection.pop(track_id, None)
                    if detection_idx in confirmed_detection_indices:
                        confirmed_detection_indices.remove(detection_idx)
                    if detection_idx in suspected_detection_indices:
                        suspected_detection_indices.remove(detection_idx)
                    print(f"[fall_detection] ✅ Recovery (Track {track_id}): person standing — red keypoints removed")
                else:
                    # Still confirmed fallen - add to confirmed list ONLY if not already there
                    # Red keypoints will show until person recovers
                    if detection_idx not in confirmed_detection_indices:
                        confirmed_detection_indices.append(detection_idx)
                    # Ensure not in suspected list (should only be in one state)
                    if detection_idx in suspected_detection_indices:
                        suspected_detection_indices.remove(detection_idx)

        # Note: We DON'T clear confirmed states after cooldown - confirmed fall persists until person recovers
        # Cooldown only prevents multiple alert events, but keypoints stay red until recovery

        # Clean up: Remove state for tracks that are no longer active (tracker handles this, but clean up our state)
        active_track_ids = {track.track_id for track in all_active_tracks}
        for track_id in list(person_states.keys()):
            if track_id not in active_track_ids:
                # Track no longer active - clear its state and VLM confirmed (so red keypoints don't stick)
                person_states.pop(track_id, None)
                fall_suspected_at.pop(track_id, None)
                fall_height.pop(track_id, None)
                self._state["fall_hip_y"].pop(track_id, None)
                self._state["fall_head_y"].pop(track_id, None)
                history.pop(track_id, None)
                if self.config_obj.vlm_enabled and self.vlm_state:
                    self.vlm_state.confirmed_falls.discard(track_id)
                    self.vlm_state.confirmed_track_to_detection.pop(track_id, None)

        # Update confirmed detection indices from VLM state: only show red for tracks that are
        # (a) still confirmed and (b) visible this frame — use current frame's detection index
        if self.config_obj.vlm_enabled and self.vlm_state:
            for track_id in list(self.vlm_state.confirmed_falls):
                detection_idx = track_to_detection_map.get(track_id)
                if detection_idx is not None:
                    if detection_idx not in confirmed_detection_indices:
                        confirmed_detection_indices.append(detection_idx)
                    if detection_idx in suspected_detection_indices:
                        suspected_detection_indices.remove(detection_idx)
                # If track not in current frame, don't add any index (red will disappear for that person)

        # Clean up: Remove any detection indices from confirmed/suspected lists if they're no longer valid
        # (This handles cases where detection indices change or persons leave the frame)
        valid_detection_indices_set = set(track_to_detection_map.values())
        confirmed_detection_indices = [idx for idx in confirmed_detection_indices if idx in valid_detection_indices_set]
        suspected_detection_indices = [idx for idx in suspected_detection_indices if idx in valid_detection_indices_set]

        # Always update state with current frame's detection indices (so keypoints show correct colors)
        # Only include indices that are actually detected in this frame
        self._state["fall_suspected_indices"] = suspected_detection_indices
        self._state["fall_confirmed_indices"] = confirmed_detection_indices

        # Emit event only when we have at least one confirmed fall and cooldown allows
        last_alert = self._state.get("last_alert_time")
        if confirmed_detection_indices:
            if last_alert is None or not isinstance(last_alert, datetime) or (now - last_alert).total_seconds() >= cooldown_sec:
                self._state["last_alert_time"] = now
                label = self.generate_label(len(confirmed_detection_indices))
                print(f"[fall_detection] 🚨 ALERT: {label} - {len(confirmed_detection_indices)} person(s) confirmed fallen")
                events.append(
                    ScenarioEvent(
                        event_type="fall_detected",
                        label=label,
                        confidence=1.0,
                        metadata={
                            "target_class": self.config_obj.target_class,
                            "fallen_count": len(confirmed_detection_indices),
                            "fallen_indices": confirmed_detection_indices,
                            "detection_method": "pose_keypoints_vlm" if self.config_obj.vlm_enabled else "pose_keypoints_timeout",
                            "alert_cooldown_seconds": cooldown_sec,
                            "vlm_enabled": self.config_obj.vlm_enabled,
                        },
                        detection_indices=confirmed_detection_indices,
                        timestamp=frame_context.timestamp,
                        frame_index=frame_context.frame_index,
                    )
                )

        return events

    def generate_label(self, count: int) -> str:
        custom_label = self.config_obj.custom_label
        if custom_label and isinstance(custom_label, str) and custom_label.strip():
            return custom_label.strip()
        if count == 1:
            return "Human fall detected"
        return f"{count} human fall(s) detected"

    def reset(self) -> None:
        self._state["history"].clear()
        self._state["baseline"].clear()
        self._state["person_states"].clear()
        self._state["fall_suspected_at"].clear()
        self._state["fall_height"].clear()
        self._state["fall_hip_y"].clear()
        self._state["fall_head_y"].clear()
        self._state["last_alert_time"] = None
        self._state["last_frame_time"] = None
        self._state["fall_suspected_indices"] = []
        self._state["fall_confirmed_indices"] = []
        if self.config_obj.vlm_enabled and self.vlm_state:
            self.vlm_state.reset()
        self._vlm_service = None
