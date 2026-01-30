"""
Restricted Zone Scenario
------------------------

Monitors a restricted zone for object presence and triggers alerts.

Behavior:
- Triggers alert when an object of specified class is INSIDE the zone
- Uses bounding box center point to determine if object is in zone
- Alert cooldown prevents rapid-fire duplicate alerts
"""

from typing import List, Dict, Any
from datetime import datetime

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from .config import RestrictedZoneConfig
from .zone_utils import is_box_in_zone


@register_scenario("restricted_zone")
class RestrictedZoneScenario(BaseScenario):
    """
    Monitors a restricted zone for object presence.
    Triggers alerts when objects of specified class are detected inside the zone.
    """

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = RestrictedZoneConfig(config, pipeline_context.task)
        self._state["last_alert_time"] = None
        self._state["objects_in_zone"] = False

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        if not self.config_obj.target_class or not self.config_obj.zone_coordinates:
            return []
        matched_indices, matched_classes = self._find_objects_in_zone(frame_context)
        if not matched_indices:
            self._state["last_alert_time"] = None
            self._state["objects_in_zone"] = False
            return []
        now = frame_context.timestamp
        last_alert_time = self._state.get("last_alert_time")
        if last_alert_time and isinstance(last_alert_time, datetime):
            if (now - last_alert_time).total_seconds() < self.config_obj.alert_cooldown_seconds:
                self._state["objects_in_zone"] = True
                return []
        label = self._generate_label(len(matched_indices))
        detections = frame_context.detections
        scores = detections.scores
        matched_scores = [scores[i] for i in matched_indices if i < len(scores)]
        avg_confidence = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0
        self._state["last_alert_time"] = now
        self._state["objects_in_zone"] = True
        return [
            ScenarioEvent(
                event_type="restricted_zone_detection",
                label=label,
                confidence=1.0,
                metadata={
                    "target_class": self.config_obj.target_class,
                    "objects_in_zone": len(matched_indices),
                    "zone_type": "polygon",
                    "alert_cooldown_seconds": self.config_obj.alert_cooldown_seconds,
                },
                detection_indices=matched_indices,
                timestamp=frame_context.timestamp,
                frame_index=frame_context.frame_index,
            )
        ]

    def _find_objects_in_zone(self, frame_context: ScenarioFrameContext) -> tuple:
        detections = frame_context.detections
        boxes = detections.boxes
        classes = detections.classes
        scores = detections.scores
        frame = frame_context.frame
        frame_height, frame_width = frame.shape[:2]
        matched_indices = []
        matched_classes = []
        target_class = self.config_obj.target_class.lower()
        confidence_threshold = self.config_obj.confidence_threshold
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if isinstance(cls, str) and cls.lower() == target_class and score >= confidence_threshold:
                if is_box_in_zone(box, self.config_obj.zone_coordinates, frame_width, frame_height):
                    matched_indices.append(i)
                    matched_classes.append(cls)
        return matched_indices, matched_classes

    def _generate_label(self, count: int) -> str:
        custom_label = self.config_obj.custom_label
        if custom_label and isinstance(custom_label, str) and custom_label.strip():
            return custom_label.strip()
        target_class = self.config_obj.target_class.replace("_", " ").title()
        if count == 1:
            return f"{target_class} detected in restricted zone"
        return f"{count} {target_class}(s) detected in restricted zone"

    def reset(self) -> None:
        self._state["last_alert_time"] = None
        self._state["objects_in_zone"] = False
