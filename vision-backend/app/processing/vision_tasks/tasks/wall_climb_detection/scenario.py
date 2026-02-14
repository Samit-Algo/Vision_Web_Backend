"""
Wall climb detection scenario
------------------------------

User polygon = wall boundary. When a person goes above the zone (detection box above wall line),
we treat it as violation: red box + alert. No orange - only red.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import logging
from datetime import datetime
from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from .config import WallClimbConfig
from .wall_zone_utils import is_box_climbing, is_box_fully_above_wall

logger = logging.getLogger(__name__)


@register_scenario("wall_climb_detection")
class WallClimbScenario(BaseScenario):
    """
    Detects when a person goes above the user-drawn wall zone.
    Any person above the zone (box top or any part above wall line) = violation → red box + alert.
    """

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = WallClimbConfig(config, pipeline_context.task)
        self._state["last_alert_time"] = None
        self._state["climbing_indices"] = []  # Unused; kept for pipeline compatibility (no orange)
        self._state["fully_above_indices"] = []
        # Centers of persons who were ever above zone → keep them red across frames
        self._state["ever_above_centers"] = set()

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        if not self.config_obj.target_class or not self.config_obj.zone_coordinates:
            self._state["climbing_indices"] = []
            self._state["fully_above_indices"] = []
            self._state["red_indices"] = []
            return []

        frame = frame_context.frame
        frame_height, frame_width = frame.shape[:2]
        detections = frame_context.detections
        boxes = detections.boxes
        classes = detections.classes
        scores = detections.scores

        target_class = self.config_obj.target_class.lower()
        conf_threshold = self.config_obj.confidence_threshold

        # Above zone = climbing (box top above wall) OR fully above (box bottom above wall). Both → violation (red).
        above_indices = []
        ever_centers = self._state.get("ever_above_centers") or set()
        ever_centers = set(ever_centers)

        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if not (isinstance(cls, str) and cls.lower() == target_class and score >= conf_threshold):
                continue
            if not box or len(box) < 4:
                continue

            climb = is_box_climbing(
                box, self.config_obj.zone_coordinates, frame_width, frame_height
            )
            fully = is_box_fully_above_wall(
                box, self.config_obj.zone_coordinates, frame_width, frame_height
            )
            above = climb or fully

            if above:
                above_indices.append(i)
                # Remember center so we keep this person red in following frames
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                max_b = max(x1, y1, x2, y2)
                if max_b <= 1.0 and frame_width > 1 and frame_height > 1:
                    x1, x2 = x1 * frame_width, x2 * frame_width
                    y1, y2 = y1 * frame_height, y2 * frame_height
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                ever_centers.add((round(cx), round(cy)))

        if len(ever_centers) > 100:
            ever_centers = set()

        # Red = anyone above this frame OR anyone whose center was ever above (same person in next frames)
        red_indices = list(above_indices)
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if i in red_indices:
                continue
            if not (isinstance(cls, str) and cls.lower() == target_class and score >= conf_threshold):
                continue
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            max_b = max(x1, y1, x2, y2)
            if max_b <= 1.0 and frame_width > 1 and frame_height > 1:
                x1, x2 = x1 * frame_width, x2 * frame_width
                y1, y2 = y1 * frame_height, y2 * frame_height
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if (round(cx), round(cy)) in ever_centers:
                red_indices.append(i)

        self._state["ever_above_centers"] = ever_centers
        self._state["climbing_indices"] = []  # No orange: only red
        self._state["fully_above_indices"] = above_indices
        self._state["red_indices"] = red_indices

        # Alert when any person is above the zone (with cooldown)
        events = []
        if above_indices:
            logger.info(
                "[wall_climb_detection] Violation: person above user zone (count=%s, indices=%s)",
                len(above_indices), above_indices,
            )
            now = frame_context.timestamp
            last_alert = self._state.get("last_alert_time")
            if last_alert is None or not isinstance(last_alert, datetime) or \
               (now - last_alert).total_seconds() >= self.config_obj.alert_cooldown_seconds:
                self._state["last_alert_time"] = now
                label = self.generate_label(len(above_indices))
                events.append(
                    ScenarioEvent(
                        event_type="wall_climb_detection",
                        label=label,
                        confidence=1.0,
                        metadata={
                            "target_class": self.config_obj.target_class,
                            "above_zone_count": len(above_indices),
                            "alert_cooldown_seconds": self.config_obj.alert_cooldown_seconds,
                        },
                        detection_indices=above_indices,
                        timestamp=frame_context.timestamp,
                        frame_index=frame_context.frame_index,
                    )
                )

        return events

    def generate_label(self, count: int) -> str:
        if self.config_obj.custom_label and isinstance(self.config_obj.custom_label, str) and self.config_obj.custom_label.strip():
            return self.config_obj.custom_label.strip()
        if count == 1:
            return "Person above wall zone"
        return f"{count} persons above wall zone"

    def reset(self) -> None:
        self._state["last_alert_time"] = None
        self._state["climbing_indices"] = []
        self._state["fully_above_indices"] = []
        self._state["red_indices"] = []
        self._state["ever_above_centers"] = set()
