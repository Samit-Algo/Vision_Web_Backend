"""
Rule: accident_presence (Generic Human Fall Detection)
------------------------------------------------------

NO USER TUNING REQUIRED.
Works with default values only.

Fall detection uses TWO methods:
1. ACT of falling: Detects the motion of falling
   - Downward hip motion (> 6 pixels/frame)
   - Body height collapse (< 70% of previous height)
   - Lying posture confirmation (angle > 50°)
   - Requires 2 consecutive frames

2. STATE of being fallen: Detects static lying position
   - Person in lying posture (angle > 50°)
   - Maintained for 5 consecutive frames
   - Useful for detecting already-fallen persons (e.g., sleeping)

Designed for:
- RTSP
- Shared memory
- Low FPS (>= 5)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

from app.processing.rule_engine.registry import register_rule


# =============================
# INTERNAL DEFAULTS (DO NOT EXPOSE)
# =============================

DEFAULT_FALL_SPEED = 6          # pixels / frame (safe for 5 FPS)
DEFAULT_HEIGHT_RATIO = 0.7      # body collapse %
DEFAULT_LYING_ANGLE = 45        # degrees (reduced from 50 for better detection)
DEFAULT_CONFIRM_FRAMES = 2      # consecutive frames for fall motion
DEFAULT_STATIC_LYING_FRAMES = 3  # frames to confirm static lying position (reduced from 5 for faster detection)


# =============================
# Utility helpers
# =============================

def _kp(person, idx) -> Optional[Tuple[float, float]]:
    """Extract keypoint at index, return (x, y) or None if invalid."""
    if idx >= len(person):
        return None
    kp = person[idx]
    if kp is None or len(kp) < 2:
        return None
    # Check confidence if available (3rd element)
    # Lower threshold to 0.25 to be less strict
    if len(kp) >= 3:
        try:
            confidence = float(kp[2])
            if confidence < 0.25:  # Lowered from 0.3 for better detection
                return None
        except (ValueError, TypeError):
            # If confidence can't be parsed, use the keypoint anyway
            pass
    return float(kp[0]), float(kp[1])


def _mid(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _angle_from_vertical(p1, p2) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _bbox_height(person) -> float:
    ys = [kp[1] for kp in person if kp and len(kp) >= 2]
    return max(ys) - min(ys) if ys else 0.0


# =============================
# Core fall analysis
# =============================

def _analyze(person, prev):
    """Analyze person pose for fall detection.
    
    Returns: (fall_motion, collapsed, lying, metrics)
    """
    # YOLO pose keypoint indices:
    # 5: left shoulder, 6: right shoulder
    # 11: left hip, 12: right hip
    ls = _kp(person, 5)
    rs = _kp(person, 6)
    lh = _kp(person, 11)
    rh = _kp(person, 12)

    if not (ls and rs and lh and rh):
        # Missing required keypoints
        if prev is None:  # Only log on first frame to avoid spam
            print(f"[_analyze] Missing keypoints - ls={ls is not None}, rs={rs is not None}, lh={lh is not None}, rh={rh is not None}")
            print(f"[_analyze] Person has {len(person)} keypoints total")
        return False, False, False, None

    shoulder_mid = _mid(ls, rs)
    hip_mid = _mid(lh, rh)

    height = _bbox_height(person)
    angle = _angle_from_vertical(shoulder_mid, hip_mid)
    
    # Debug: Log keypoint positions for first frame
    if prev is None:
        print(f"[_analyze] First frame - shoulders: {shoulder_mid}, hips: {hip_mid}, height: {height:.1f}px, angle: {angle:.1f}°")

    fall_motion = False
    collapsed = False

    if prev:
        dy = hip_mid[1] - prev["hip_y"]
        if dy > DEFAULT_FALL_SPEED:
            fall_motion = True

        if prev["height"] > 0:
            if height / prev["height"] < DEFAULT_HEIGHT_RATIO:
                collapsed = True

    lying = angle > DEFAULT_LYING_ANGLE

    metrics = {
        "hip_y": hip_mid[1],
        "height": height,
        "angle": angle,
    }

    return fall_motion, collapsed, lying, metrics


# =============================
# Rule entry point
# =============================

@register_rule("accident_presence")
def evaluate_accident_presence(
    rule: Dict[str, Any],
    detections: Dict[str, Any],
    task: Dict[str, Any],
    rule_state: Dict[str, Any],
    now: datetime,
) -> Optional[Dict[str, Any]]:

    # Debug: print entering rule
    print(f"[evaluate_accident_presence] ========== RULE INVOKED ==========")
    print(f"[evaluate_accident_presence] Rule: {rule}")
    print(f"[evaluate_accident_presence] Detections keys: {list(detections.keys())}")
    print(f"[evaluate_accident_presence] Classes: {detections.get('classes', [])}")
    print(f"[evaluate_accident_presence] Keypoints present: {'keypoints' in detections}")
    if 'keypoints' in detections:
        kp_data = detections.get('keypoints', [])
        print(f"[evaluate_accident_presence] Keypoints count: {len(kp_data)}")

    # ---------------------------------
    # 1. Class filter (VERY IMPORTANT)
    # ---------------------------------
    target_class = (rule.get("class")).lower()
    classes = detections.get("classes") or []

    print(f"[evaluate_accident_presence] Looking for class '{target_class}'. Detected classes: {classes}")

    if target_class not in classes:
        print(f"[evaluate_accident_presence] Target class '{target_class}' not found. Exiting.")
        return None

    keypoints = detections.get("keypoints") or []
    print(f"[evaluate_accident_presence] Keypoints received: {len(keypoints)} person(s)")
    if not keypoints:
        print(f"[evaluate_accident_presence] No keypoints found. Clearing rule_state and exiting.")
        rule_state.clear()
        return None
    
    # Debug: Print keypoint structure for first person
    if keypoints and len(keypoints) > 0:
        first_person = keypoints[0]
        print(f"[evaluate_accident_presence] First person keypoints: {len(first_person)} points")
        if len(first_person) > 0:
            print(f"[evaluate_accident_presence] First keypoint sample: {first_person[0] if first_person[0] else 'None'}")

    history = rule_state.setdefault("history", {})
    fall_motion_counters = rule_state.setdefault("fall_motion_counters", {})
    static_lying_counters = rule_state.setdefault("static_lying_counters", {})

    # Clean up state for persons no longer in frame
    current_indices = set(range(len(keypoints)))
    for idx in list(history.keys()):
        if idx not in current_indices:
            history.pop(idx, None)
            fall_motion_counters.pop(idx, None)
            static_lying_counters.pop(idx, None)

    fallen_ids = []

    for idx, person in enumerate(keypoints):
        prev = history.get(idx)

        fall_motion, collapsed, lying, metrics = _analyze(person, prev)

        print(f"[evaluate_accident_presence] Person {idx}: fall_motion={fall_motion}, collapsed={collapsed}, lying={lying}, metrics={metrics}")

        if metrics:
            history[idx] = metrics
        else:
            # No valid metrics, skip this person
            print(f"[evaluate_accident_presence] Person {idx}: No valid metrics (missing keypoints), skipping")
            continue

        # Method 1: Detect the ACT of falling (motion + collapse + lying)
        # Each person has their own counter
        if fall_motion and collapsed and lying:
            fall_motion_counters[idx] = fall_motion_counters.get(idx, 0) + 1
            print(f"[evaluate_accident_presence] Person {idx}: Fall motion detected! Counter: {fall_motion_counters[idx]}")
        else:
            fall_motion_counters[idx] = max(0, fall_motion_counters.get(idx, 0) - 1)

        # Method 2: Detect the STATE of being fallen (static lying position)
        # If person is lying down and has been in that state for multiple frames
        height = metrics.get("height", 0)
        angle = metrics.get("angle", 0)
        
        print(f"[evaluate_accident_presence] Person {idx}: lying={lying}, angle={angle:.1f}°, height={height:.1f}px, threshold={DEFAULT_LYING_ANGLE}°")
        
        if lying:
            # Person is in lying posture - check if we have valid keypoints
            # Height > 20 ensures we have valid keypoints (not noise)
            if height > 20:
                static_lying_counters[idx] = static_lying_counters.get(idx, 0) + 1
                print(f"[evaluate_accident_presence] Person {idx}: Static lying confirmed! Counter: {static_lying_counters[idx]}/{DEFAULT_STATIC_LYING_FRAMES}")
            else:
                # Height too small, might be noise or partial detection
                static_lying_counters[idx] = max(0, static_lying_counters.get(idx, 0) - 1)
                print(f"[evaluate_accident_presence] Person {idx}: Lying but height too small ({height:.1f}px), decrementing counter")
        else:
            # Not lying, reset counter for this person
            if static_lying_counters.get(idx, 0) > 0:
                print(f"[evaluate_accident_presence] Person {idx}: Not lying (angle={angle:.1f}°), resetting static counter")
            static_lying_counters[idx] = 0

        current_fall_motion = fall_motion_counters.get(idx, 0)
        current_static_lying = static_lying_counters.get(idx, 0)
        print(f"[evaluate_accident_presence] Person {idx}: fall_motion_counter={current_fall_motion}/{DEFAULT_CONFIRM_FRAMES}, static_lying_counter={current_static_lying}/{DEFAULT_STATIC_LYING_FRAMES}")

        # Trigger if either:
        # 1. Fall motion detected (act of falling) - requires motion + collapse + lying
        # 2. Static lying position maintained (state of being fallen) - just lying for N frames
        if current_fall_motion >= DEFAULT_CONFIRM_FRAMES:
            print(f"[evaluate_accident_presence] 🚨 Person {idx}: FALL DETECTED via motion!")
            fallen_ids.append(idx)
        elif current_static_lying >= DEFAULT_STATIC_LYING_FRAMES:
            print(f"[evaluate_accident_presence] 🚨 Person {idx}: FALL DETECTED via static lying!")
            fallen_ids.append(idx)

    print(f"[evaluate_accident_presence] Fallen IDs: {fallen_ids}")

    if not fallen_ids:
        print(f"[evaluate_accident_presence] No confirmed fall detected.")
        return None

    # ---------------------------------
    # FALL CONFIRMED
    # ---------------------------------
    print(f"[evaluate_accident_presence] 🚨 Fall detected! Returning alert event.")
    
    # Compute detection indices: accident_presence matches "person" class detections
    # that have valid keypoints (pose data). Map fallen_ids (keypoint indices) to
    # detection indices by matching person keypoints to person class detections.
    detected_classes_list = detections.get("classes") or []
    matched_indices: List[int] = []
    # For accident_presence, we match person detections that have pose keypoints
    # The fallen_ids are keypoint indices, but we need to map to detection indices
    # Since keypoints and classes are aligned (one person = one detection), use fallen_ids directly
    for kp_idx in fallen_ids:
        if kp_idx < len(detected_classes_list):
            detected_class = detected_classes_list[kp_idx]
            if isinstance(detected_class, str) and detected_class.lower() == "person":
                matched_indices.append(kp_idx)
    
    return {
        "label": "🚨 Human fall detected",
        "fallen_count": len(fallen_ids),
        "fallen_indices": fallen_ids,
        "matched_detection_indices": matched_indices,
    }