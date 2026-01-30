"""
Fall Detection Scenario
-----------------------

Detects human falls using pose keypoint analysis.

Algorithm:
1. Monitors hip position for sudden downward movement
2. Tracks body height collapse
3. Detects lying posture using body angle
4. Confirms fall after N consecutive frames

Designed for:
- YOLOv8 Pose models
- Live stream / RTSP
- FPS >= 5
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import math

from app.processing.scenarios.contracts import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent
)
from app.processing.scenarios.registry import register_scenario
from app.processing.scenarios.fall_detection.config import FallDetectionConfig


# =============================
# KEYPOINT HELPERS
# =============================

def _get_keypoint(person_keypoints: List[List[float]], idx: int, 
                  confidence_threshold: float = 0.3) -> Optional[Tuple[float, float]]:
    """
    Extract keypoint coordinates if available and confident.
    
    Args:
        person_keypoints: List of keypoints for one person [[x, y, conf], ...]
        idx: Keypoint index (COCO format: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip)
        confidence_threshold: Minimum confidence to consider keypoint valid
    
    Returns:
        (x, y) tuple if valid, None otherwise
    """
    if idx >= len(person_keypoints):
        return None
    
    kp = person_keypoints[idx]
    if not kp or len(kp) < 2:
        return None
    
    # Check confidence if available (3rd element)
    if len(kp) >= 3 and kp[2] < confidence_threshold:
        return None
    
    return float(kp[0]), float(kp[1])


def _midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate midpoint between two points."""
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)


def _angle_from_vertical(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate angle of line from vertical (0 = vertical, 90 = horizontal).
    
    Args:
        p1: First point (typically shoulder midpoint)
        p2: Second point (typically hip midpoint)
    
    Returns:
        Angle in degrees (0-90)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    if dx == 0 and dy == 0:
        return 0.0
    
    # Calculate angle from vertical
    # atan2(abs(dx), abs(dy)) gives angle from vertical axis
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)


def _calculate_bbox_height(person_keypoints: List[List[float]]) -> float:
    """
    Calculate bounding box height from keypoints.
    
    Args:
        person_keypoints: List of keypoints for one person
    
    Returns:
        Height in pixels
    """
    valid_ys = []
    for kp in person_keypoints:
        if kp and len(kp) >= 2:
            valid_ys.append(kp[1])
    
    if not valid_ys:
        return 0.0
    
    return max(valid_ys) - min(valid_ys)


# =============================
# CORE FALL ANALYSIS
# =============================

def _analyze_person_fall(
    person_keypoints: List[List[float]],
    prev_metrics: Optional[Dict[str, Any]] = None,
    config: FallDetectionConfig = None
) -> Tuple[bool, bool, Optional[Dict[str, Any]]]:
    """
    Analyze person pose to detect falling motion and lying posture.
    
    Args:
        person_keypoints: List of keypoints for one person
        prev_metrics: Previous frame metrics (for motion detection)
        config: Configuration object
    
    Returns:
        (falling: bool, lying: bool, metrics: dict or None)
    """
    if config is None:
        return False, False, None
    
    # COCO keypoint indices:
    # 5 = left_shoulder, 6 = right_shoulder
    # 11 = left_hip, 12 = right_hip
    
    left_shoulder = _get_keypoint(person_keypoints, 5, config.kp_confidence_threshold)
    right_shoulder = _get_keypoint(person_keypoints, 6, config.kp_confidence_threshold)
    left_hip = _get_keypoint(person_keypoints, 11, config.kp_confidence_threshold)
    right_hip = _get_keypoint(person_keypoints, 12, config.kp_confidence_threshold)
    
    # Need at least shoulders and hips to analyze
    if not (left_shoulder and right_shoulder and left_hip and right_hip):
        return False, False, None
    
    # Calculate midpoints
    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    
    # Calculate metrics
    height = _calculate_bbox_height(person_keypoints)
    angle = _angle_from_vertical(shoulder_mid, hip_mid)
    
    # Detect lying posture:
    # - High angle (body horizontal) OR
    # - Very small height (person collapsed)
    lying = (angle > config.lying_angle_threshold or 
             height < config.min_height_for_standing)
    
    falling = False
    
    # Detect falling motion by comparing with previous frame
    if prev_metrics:
        hip_drop = hip_mid[1] - prev_metrics["hip_y"]  # Positive = downward movement
        height_drop = prev_metrics["height"] - height  # Positive = height decreased
        prev_angle = prev_metrics.get("angle", 0)
        
        # Method 1: Sudden hip drop + significant height collapse
        if (hip_drop > config.hip_drop_threshold and 
            height_drop > (config.height_drop_ratio * prev_metrics["height"])):
            falling = True
        
        # Method 2: Transition from standing to lying with height drop
        # If was standing (low angle) and now lying (high angle)
        if (prev_angle <= config.lying_angle_threshold and 
            angle > config.lying_angle_threshold):
            # And there's significant height drop
            if height_drop > (config.height_drop_ratio * prev_metrics["height"]):
                falling = True
    
    metrics = {
        "hip_y": hip_mid[1],
        "height": height,
        "angle": angle,
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
    }
    
    return falling, lying, metrics


# =============================
# SCENARIO CLASS
# =============================

@register_scenario("fall_detection")
class FallDetectionScenario(BaseScenario):
    """
    Detects human falls using pose keypoint analysis.
    
    Monitors person pose over time to detect:
    - Sudden downward movement (hip drop)
    - Body height collapse
    - Lying posture (horizontal body angle)
    """
    
    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        
        # Load configuration
        self.config_obj = FallDetectionConfig(config, pipeline_context.task)
        
        # Initialize state
        # history: per-person metrics from previous frames
        # fall_counter: per-person counter for consecutive fall frames
        self._state["history"] = {}  # {person_idx: metrics_dict}
        self._state["fall_counter"] = {}  # {person_idx: int}
        self._state["last_alert_time"] = None
    
    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process a single frame for fall detection.
        
        Returns:
            List of ScenarioEvent if fall detected, empty list otherwise
        """
        # Early exit if disabled
        if not self.config_obj.enabled:
            return []
        
        # Early exit if no target class
        if not self.config_obj.target_class:
            return []
        
        detections = frame_context.detections
        
        # Filter for target class
        target_class = self.config_obj.target_class
        matched_indices = []
        
        for i, cls in enumerate(detections.classes):
            if isinstance(cls, str) and cls.lower() == target_class:
                # Check confidence threshold
                if i < len(detections.scores) and detections.scores[i] >= self.config_obj.confidence_threshold:
                    matched_indices.append(i)
        
        if not matched_indices:
            # No persons detected, clear state
            self._state["history"].clear()
            self._state["fall_counter"].clear()
            return []
        
        # Check alert cooldown
        now = frame_context.timestamp
        last_alert_time = self._state.get("last_alert_time")
        
        if last_alert_time and isinstance(last_alert_time, datetime):
            elapsed = (now - last_alert_time).total_seconds()
            if elapsed < self.config_obj.alert_cooldown_seconds:
                # Still in cooldown, continue monitoring but don't alert
                pass
        
        # Get keypoints
        keypoints = detections.keypoints
        if not keypoints:
            return []
        
        # Analyze each person
        fallen_person_indices = []
        history = self._state["history"]
        fall_counter = self._state["fall_counter"]
        
        for person_idx in matched_indices:
            if person_idx >= len(keypoints):
                continue
            
            person_keypoints = keypoints[person_idx]
            if not person_keypoints:
                continue
            
            # Get previous metrics for this person
            prev_metrics = history.get(person_idx)
            
            # Analyze current frame
            falling, lying, metrics = _analyze_person_fall(
                person_keypoints,
                prev_metrics,
                self.config_obj
            )
            
            # Update history
            if metrics:
                history[person_idx] = metrics
            
            # Debug logging
            print(
                f"[FALL_DETECTION] Person {person_idx}: "
                f"falling={falling}, lying={lying}, "
                f"angle={metrics['angle']:.1f}°, height={metrics['height']:.1f}px"
            )
            
            # Update fall counter
            # If lying posture detected, increment counter
            # If also detected falling motion, increment faster
            if lying:
                if falling:
                    # Falling motion + lying = strong fall signal
                    fall_counter[person_idx] = fall_counter.get(person_idx, 0) + 2
                else:
                    # Just lying (static) = slower confirmation
                    fall_counter[person_idx] = fall_counter.get(person_idx, 0) + 1
            else:
                # Not lying, reset counter
                fall_counter[person_idx] = 0
            
            # Check if fall is confirmed
            if fall_counter.get(person_idx, 0) >= self.config_obj.confirm_frames:
                fallen_person_indices.append(person_idx)
                print(
                    f"[FALL_DETECTION] 🚨 FALL CONFIRMED for Person {person_idx} "
                    f"(counter={fall_counter[person_idx]})"
                )
        
        # Generate alert if any falls detected
        if not fallen_person_indices:
            return []
        
        # Check cooldown again before alerting
        if last_alert_time and isinstance(last_alert_time, datetime):
            elapsed = (now - last_alert_time).total_seconds()
            if elapsed < self.config_obj.alert_cooldown_seconds:
                # Still in cooldown, don't alert
                return []
        
        # Generate alert label
        label = self._generate_label(len(fallen_person_indices))
        
        # Print alert to console
        print("=" * 60)
        print(f"🚨 [FALL DETECTION ALERT] {label}")
        print(f"   👤 {len(fallen_person_indices)} person(s) detected falling")
        print(f"   📍 Person indices: {fallen_person_indices}")
        print(f"   🕐 Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📹 Frame index: {frame_context.frame_index}")
        print("=" * 60)
        
        # Update last alert time
        self._state["last_alert_time"] = now
        
        # Create event
        event = ScenarioEvent(
            event_type="fall_detected",
            label=label,
            confidence=1.0,
            metadata={
                "target_class": self.config_obj.target_class,
                "fallen_count": len(fallen_person_indices),
                "fallen_indices": fallen_person_indices,
                "detection_method": "pose_keypoints",
                "alert_cooldown_seconds": self.config_obj.alert_cooldown_seconds,
            },
            detection_indices=fallen_person_indices,
            timestamp=frame_context.timestamp,
            frame_index=frame_context.frame_index
        )
        
        return [event]
    
    def _generate_label(self, count: int) -> str:
        """Generate alert label."""
        custom_label = self.config_obj.custom_label
        
        if custom_label and isinstance(custom_label, str) and custom_label.strip():
            return custom_label.strip()
        
        # Default label
        if count == 1:
            return "🚨 Human fall detected"
        else:
            return f"🚨 {count} human fall(s) detected"
    
    def reset(self) -> None:
        """Reset scenario state."""
        self._state["history"].clear()
        self._state["fall_counter"].clear()
        self._state["last_alert_time"] = None
