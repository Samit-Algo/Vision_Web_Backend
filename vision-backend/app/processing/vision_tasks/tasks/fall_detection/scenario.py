"""
Fall detection scenario
------------------------

Uses pose keypoints to detect human falls: hip drop, height collapse, lying angle.
Confirms after N consecutive frames; alert cooldown limits repeats. Needs YOLO pose model.
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

from .config import FallDetectionConfig

# -----------------------------------------------------------------------------
# Keypoint helpers (shoulder, hip, angle, height)
# -----------------------------------------------------------------------------

def get_keypoint(
    person_keypoints: List[List[float]],
    idx: int,
    confidence_threshold: float = 0.3,
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
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return 0.0
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)


def calculate_bbox_height(person_keypoints: List[List[float]]) -> float:
    valid_ys = [kp[1] for kp in person_keypoints if kp and len(kp) >= 2]
    if not valid_ys:
        return 0.0
    return max(valid_ys) - min(valid_ys)


def analyze_person_fall(
    person_keypoints: List[List[float]],
    prev_metrics: Optional[Dict[str, Any]] = None,
    config: Optional[FallDetectionConfig] = None,
) -> Tuple[bool, bool, Optional[Dict[str, Any]]]:
    if config is None:
        return False, False, None
    left_shoulder = get_keypoint(person_keypoints, 5, config.kp_confidence_threshold)
    right_shoulder = get_keypoint(person_keypoints, 6, config.kp_confidence_threshold)
    left_hip = get_keypoint(person_keypoints, 11, config.kp_confidence_threshold)
    right_hip = get_keypoint(person_keypoints, 12, config.kp_confidence_threshold)
    if not (left_shoulder and right_shoulder and left_hip and right_hip):
        return False, False, None
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    height = calculate_bbox_height(person_keypoints)
    angle = angle_from_vertical(shoulder_mid, hip_mid)
    lying = (
        angle > config.lying_angle_threshold
        or height < config.min_height_for_standing
    )
    falling = False
    if prev_metrics:
        hip_drop = hip_mid[1] - prev_metrics["hip_y"]
        height_drop = prev_metrics["height"] - height
        prev_angle = prev_metrics.get("angle", 0)
        if hip_drop > config.hip_drop_threshold and height_drop > (
            config.height_drop_ratio * prev_metrics["height"]
        ):
            falling = True
        if (
            prev_angle <= config.lying_angle_threshold
            and angle > config.lying_angle_threshold
            and height_drop > (config.height_drop_ratio * prev_metrics["height"])
        ):
            falling = True
    metrics = {
        "hip_y": hip_mid[1],
        "height": height,
        "angle": angle,
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
    }
    return falling, lying, metrics


# -----------------------------------------------------------------------------
# Scenario
# -----------------------------------------------------------------------------


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
        self.config_obj = FallDetectionConfig(config, pipeline_context.task)
        self._state["history"] = {}
        self._state["fall_counter"] = {}
        self._state["last_alert_time"] = None

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        # Step 1: Must be enabled with target class (person)
        if not self.config_obj.enabled or not self.config_obj.target_class:
            return []
        detections = frame_context.detections
        target_class = self.config_obj.target_class
        matched_indices = []
        for i, cls in enumerate(detections.classes):
            if isinstance(cls, str) and cls.lower() == target_class:
                if i < len(detections.scores) and detections.scores[i] >= self.config_obj.confidence_threshold:
                    matched_indices.append(i)
        if not matched_indices:
            self._state["history"].clear()
            self._state["fall_counter"].clear()
            return []
        now = frame_context.timestamp
        last_alert_time = self._state.get("last_alert_time")
        keypoints = detections.keypoints
        if not keypoints:
            return []
        # Step 2: For each matched person, analyze pose (falling/lying) and update counters
        fallen_person_indices = []
        history = self._state["history"]
        fall_counter = self._state["fall_counter"]
        for person_idx in matched_indices:
            if person_idx >= len(keypoints):
                continue
            person_keypoints = keypoints[person_idx]
            if not person_keypoints:
                continue
            prev_metrics = history.get(person_idx)
            falling, lying, metrics = analyze_person_fall(
                person_keypoints, prev_metrics, self.config_obj
            )
            if metrics:
                history[person_idx] = metrics
            if lying:
                if falling:
                    fall_counter[person_idx] = fall_counter.get(person_idx, 0) + 2
                else:
                    fall_counter[person_idx] = fall_counter.get(person_idx, 0) + 1
            else:
                fall_counter[person_idx] = 0
            if fall_counter.get(person_idx, 0) >= self.config_obj.confirm_frames:
                fallen_person_indices.append(person_idx)
        if not fallen_person_indices:
            return []
        # Step 3: Cooldown check, then emit event
        if last_alert_time and isinstance(last_alert_time, datetime):
            if (now - last_alert_time).total_seconds() < self.config_obj.alert_cooldown_seconds:
                return []
        label = self.generate_label(len(fallen_person_indices))
        self._state["last_alert_time"] = now
        return [
            ScenarioEvent(
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
                frame_index=frame_context.frame_index,
            )
        ]

    def generate_label(self, count: int) -> str:
        custom_label = self.config_obj.custom_label
        if custom_label and isinstance(custom_label, str) and custom_label.strip():
            return custom_label.strip()
        if count == 1:
            return "Human fall detected"
        return f"{count} human fall(s) detected"

    def reset(self) -> None:
        self._state["history"].clear()
        self._state["fall_counter"].clear()
        self._state["last_alert_time"] = None
