"""
Fall detection scenario
------------------------

Real-time fall detection using a state machine and FPS-independent metrics:

  NORMAL → FALL_SUSPECTED → (RECOVERED → NORMAL) or (TIMEOUT → CONFIRMED_FALL → ALERT)

- Fall suspected: sudden downward movement (A) + (height collapse (B) or horizontal torso (C)).
- Recovery: height increased, torso vertical, or hip/head moved up within recovery window.
- Confirmed fall: no recovery within recovery_timeout_seconds → emit alert.

Keypoints: FALL_SUSPECTED → orange, CONFIRMED_FALL → red (set in state for pipeline/UI).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import math
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
    
    # 1. Height recovery: person is standing again
    if current_height >= fall_height * config.recovery_height_ratio:
        return True
    
    # 2. Torso vertical: person is upright (but might still be on ground)
    torso_vertical = current_angle <= config.recovery_torso_angle_max
    
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


@register_scenario("fall_detection")
class FallDetectionScenario(BaseScenario):
    """
    Detects human falls using a state machine and FPS-independent pose analysis.

    - FALL SUSPECTED: sudden drop + (height collapse or horizontal torso).
    - Recovery window: recovery_timeout_seconds (configurable, default 3 s).
    - CONFIRMED FALL: no recovery within window → alert; keypoints orange (suspected), red (confirmed).
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

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        if not self.config_obj.enabled or not self.config_obj.target_class:
            self._state["fall_suspected_indices"] = []
            self._state["fall_confirmed_indices"] = []
            return []

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
        recovery_timeout = self.config_obj.recovery_timeout_seconds

        suspected_detection_indices = []  # Detection indices for drawing (orange)
        confirmed_detection_indices = []  # Detection indices for drawing (red)
        events = []

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
                
                if fall_detected:
                    person_states[track_id] = STATE_FALL_SUSPECTED
                    fall_suspected_at[track_id] = now
                    fall_height[track_id] = metrics["height"]
                    self._state["fall_hip_y"][track_id] = metrics["hip_y"]
                    self._state["fall_head_y"][track_id] = metrics.get("head_y")
                    suspected_detection_indices.append(detection_idx)

            elif state == STATE_FALL_SUSPECTED:
                suspected_detection_indices.append(detection_idx)
                fall_time = fall_suspected_at.get(track_id)
                stored_fall_height = fall_height.get(track_id) or metrics["height"]
                stored_fall_hip_y = self._state["fall_hip_y"].get(track_id, metrics["hip_y"])
                stored_fall_head_y = self._state["fall_head_y"].get(track_id)

                if check_recovery(metrics, stored_fall_height, stored_fall_hip_y, stored_fall_head_y, self.config_obj):
                    # Recovered → back to NORMAL
                    person_states[track_id] = STATE_NORMAL
                    fall_suspected_at.pop(track_id, None)
                    fall_height.pop(track_id, None)
                    self._state["fall_hip_y"].pop(track_id, None)
                    self._state["fall_head_y"].pop(track_id, None)
                    if detection_idx in suspected_detection_indices:
                        suspected_detection_indices.remove(detection_idx)
                elif fall_time is not None and isinstance(now, datetime) and isinstance(fall_time, datetime):
                    elapsed = (now - fall_time).total_seconds()
                    if elapsed >= recovery_timeout:
                        # Timeout → CONFIRMED_FALL
                        person_states[track_id] = STATE_CONFIRMED_FALL
                        fall_suspected_at.pop(track_id, None)
                        fall_height.pop(track_id, None)
                        self._state["fall_hip_y"].pop(track_id, None)
                        self._state["fall_head_y"].pop(track_id, None)
                        if detection_idx in suspected_detection_indices:
                            suspected_detection_indices.remove(detection_idx)
                        confirmed_detection_indices.append(detection_idx)
                        print(f"[fall_detection] 🔴 CONFIRMED FALL (Track {track_id}): No recovery after {elapsed:.1f}s")

            elif state == STATE_CONFIRMED_FALL:
                # Person is confirmed fallen - check for recovery
                stored_fall_height = fall_height.get(track_id) or metrics["height"]
                stored_fall_hip_y = self._state["fall_hip_y"].get(track_id, metrics["hip_y"])
                stored_fall_head_y = self._state["fall_head_y"].get(track_id)
                
                if check_recovery(metrics, stored_fall_height, stored_fall_hip_y, stored_fall_head_y, self.config_obj):
                    # Person recovered - clear confirmed state and return to NORMAL
                    person_states[track_id] = STATE_NORMAL
                    fall_height.pop(track_id, None)
                    self._state["fall_hip_y"].pop(track_id, None)
                    self._state["fall_head_y"].pop(track_id, None)
                    # Explicitly remove from confirmed list (red keypoints should disappear)
                    if detection_idx in confirmed_detection_indices:
                        confirmed_detection_indices.remove(detection_idx)
                    # Also ensure not in suspected list
                    if detection_idx in suspected_detection_indices:
                        suspected_detection_indices.remove(detection_idx)
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
                # Track no longer active - clear its state
                person_states.pop(track_id, None)
                fall_suspected_at.pop(track_id, None)
                fall_height.pop(track_id, None)
                self._state["fall_hip_y"].pop(track_id, None)
                self._state["fall_head_y"].pop(track_id, None)
                history.pop(track_id, None)
                # Note: We keep baseline for a bit in case person comes back

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
                            "detection_method": "pose_keypoints",
                            "alert_cooldown_seconds": cooldown_sec,
                            "recovery_timeout_seconds": recovery_timeout,
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
