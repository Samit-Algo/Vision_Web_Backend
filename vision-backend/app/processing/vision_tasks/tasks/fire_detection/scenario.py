"""
Fire Detection Scenario
-----------------------

Detects fire/flames in camera feed using a fine-tuned YOLO model.
Triggers critical alerts when fire is detected.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from .config import FireDetectionConfig


def _is_point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            else:
                xinters = p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _is_box_in_zone(
    box: List[float],
    zone_coords: List[List[float]],
    frame_width: int,
    frame_height: int,
) -> bool:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_x_norm = center_x / frame_width
    center_y_norm = center_y / frame_height
    return _is_point_in_polygon((center_x_norm, center_y_norm), zone_coords)


@register_scenario("fire_detection")
class FireDetectionScenario(BaseScenario):
    """
    Detects fire/flames in camera feed.
    Triggers critical alerts when fire is confirmed across multiple frames.
    """

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = FireDetectionConfig(config, pipeline_context.task)
        self._state["last_alert_time"] = None
        self._state["fire_detected"] = False
        self._state["consecutive_fire_frames"] = 0
        self._state["total_fire_detections"] = 0
        self._state["fire_detection_history"] = []
        self._state["detection_counts"] = defaultdict(int)

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        fire_detections = self._find_fire_detections(frame_context)
        if not fire_detections:
            if self._state["consecutive_fire_frames"] > 0:
                self._state["consecutive_fire_frames"] -= 1
            if self._state["consecutive_fire_frames"] == 0:
                self._state["fire_detected"] = False
            return []
        self._state["consecutive_fire_frames"] += 1
        self._state["total_fire_detections"] += 1
        matched_indices, matched_classes, matched_scores = fire_detections
        avg_confidence = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0
        max_confidence = max(matched_scores) if matched_scores else 0.0
        self._state["fire_detected"] = True
        now = frame_context.timestamp
        is_confirmed = self._state["consecutive_fire_frames"] >= self.config_obj.confirm_frames
        is_alert = False
        if is_confirmed:
            last_alert_time = self._state.get("last_alert_time")
            in_cooldown = (
                last_alert_time is not None
                and isinstance(last_alert_time, datetime)
                and (now - last_alert_time).total_seconds() < self.config_obj.alert_cooldown_seconds
            )
            if not in_cooldown:
                is_alert = True
                self._state["last_alert_time"] = now
                for cls in matched_classes:
                    self._state["detection_counts"][cls] += 1
                self._state["fire_detection_history"].append({
                    "timestamp": now.isoformat(),
                    "frame_index": frame_context.frame_index,
                    "classes": matched_classes,
                    "scores": matched_scores,
                })
                if len(self._state["fire_detection_history"]) > 100:
                    self._state["fire_detection_history"] = self._state["fire_detection_history"][-100:]
        if is_alert:
            label = self._generate_label(len(matched_indices), matched_classes)
            alert_type = "critical"
            severity = "critical"
        else:
            label = f"Fire: {len(matched_indices)} detection(s)"
            alert_type = "detection"
            severity = "warning" if is_confirmed else "info"
        return [
            ScenarioEvent(
                event_type="fire_detection",
                label=label,
                confidence=max_confidence,
                metadata={
                    "alert_type": alert_type,
                    "severity": severity,
                    "is_alert": is_alert,
                    "is_confirmed": is_confirmed,
                    "fire_detected": True,
                    "detected_classes": matched_classes,
                    "detection_count": len(matched_indices),
                    "avg_confidence": avg_confidence,
                    "max_confidence": max_confidence,
                    "consecutive_frames": self._state["consecutive_fire_frames"],
                    "total_detections": self._state["total_fire_detections"],
                    "zone_applied": self.config_obj.zone_applied,
                    "alert_cooldown_seconds": self.config_obj.alert_cooldown_seconds,
                },
                detection_indices=matched_indices,
                timestamp=frame_context.timestamp,
                frame_index=frame_context.frame_index,
            )
        ]

    def _find_fire_detections(
        self, frame_context: ScenarioFrameContext
    ) -> Optional[Tuple[List[int], List[str], List[float]]]:
        detections = frame_context.detections
        boxes = detections.boxes
        classes = detections.classes
        scores = detections.scores
        frame = frame_context.frame
        frame_height, frame_width = frame.shape[:2]
        matched_indices = []
        matched_classes = []
        matched_scores = []
        target_classes = [c.lower() for c in self.config_obj.target_classes]
        confidence_threshold = self.config_obj.confidence_threshold
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if isinstance(cls, str):
                cls_lower = cls.lower()
                is_fire_class = any(
                    target in cls_lower or cls_lower in target for target in target_classes
                )
                if is_fire_class and score >= confidence_threshold:
                    if self.config_obj.zone_applied and self.config_obj.zone_coordinates:
                        if not _is_box_in_zone(
                            box, self.config_obj.zone_coordinates, frame_width, frame_height
                        ):
                            continue
                    matched_indices.append(i)
                    matched_classes.append(cls)
                    matched_scores.append(float(score))
        if matched_indices:
            return matched_indices, matched_classes, matched_scores
        return None

    def _generate_label(self, count: int, classes: List[str]) -> str:
        custom_label = self.config_obj.custom_label
        if custom_label and isinstance(custom_label, str) and custom_label.strip():
            return custom_label.strip()
        unique_classes = list(set(classes))
        if "smoke" in [c.lower() for c in unique_classes]:
            if "fire" in [c.lower() for c in unique_classes] or "flame" in [c.lower() for c in unique_classes]:
                return "FIRE AND SMOKE DETECTED!"
            return "SMOKE DETECTED - Possible fire!"
        if count == 1:
            return "FIRE DETECTED!"
        return f"FIRE DETECTED - {count} fire sources!"

    def reset(self) -> None:
        self._state["last_alert_time"] = None
        self._state["fire_detected"] = False
        self._state["consecutive_fire_frames"] = 0
        self._state["total_fire_detections"] = 0
        self._state["fire_detection_history"] = []
        self._state["detection_counts"] = defaultdict(int)
