"""
Wall Climb Detection - Scenario
-------------------------------

- User draws a polygon for the wall boundary.
- Orange: person is climbing (part of box above wall).
- Red: person is fully above the wall. Once red, we keep the color red.
"""

from typing import List, Dict, Any
from datetime import datetime

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from .config import WallClimbConfig
from .wall_zone_utils import is_box_climbing, is_box_fully_above_wall


@register_scenario("wall_climb_detection")
class WallClimbScenario(BaseScenario):
    """
    Detects when a person climbs or is fully above a wall boundary.
    Orange = climbing (part above wall). Red = fully above (and stays red).
    """

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = WallClimbConfig(config, pipeline_context.task)
        self._state["last_alert_time"] = None
        # Indices (into merged detections) that are climbing this frame -> orange
        self._state["climbing_indices"] = []
        # Indices that are fully above this frame
        self._state["fully_above_indices"] = []
        # Once a detection is fully above, we keep it red (persistent set of indices)
        # We store by a simple key: we'll use (box center or box hash) - actually we just keep
        # indices per frame; pipeline will map "ever fully above" by tracking over time.
        # Simple approach: ever_fully_above_indices = set of indices that have ever been fully above.
        # But indices change every frame (different detections). So we need to track "which boxes"
        # have been fully above. Easiest: store the last known "fully_above_indices" and treat
        # "ever red" as: if current frame this index is fully above, add to ever set; when we
        # build red list for UI we send: red = ever_fully_above (current frame mapping).
        # So we need to persist "ever fully above" by something stable. Option A: by index in
        # filtered list we can't. Option B: by track_id if we had tracking. Option C: by position
        # - if a box is fully above this frame, we add its index to a set that we pass as "red";
        # next frame we don't have same index for same person. So we need to match "same person"
        # across frames. Without tracking, simplest is: red = fully_above_indices this frame only;
        # "once red stay red" = once we have emitted an event for "someone fully above", we
        # keep showing red for any detection that is currently fully above. So we don't need
        # persistent ever_fully_above per detection; we need: this frame, who is climbing (orange)
        # and who is fully above (red). "Once red stay red" for THAT person: if we had tracking
        # we'd mark that track_id as "ever red". Without tracking, we can approximate: keep a
        # set of (normalized box center rounded) that have been fully above; if a detection's
        # center is in that set, treat as red. So ever_fully_above_centers = set of (cx, cy) rounded.
        # When we see a box fully above, add (round(cx), round(cy)) to ever set. When building
        # red list: if box is fully above OR its center is in ever_fully_above_centers, red.
        # Let me do that.
        self._state["ever_fully_above_centers"] = set()  # set of (round(cx), round(cy)) in pixel

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        if not self.config_obj.target_class or not self.config_obj.zone_coordinates:
            self._state["climbing_indices"] = []
            self._state["fully_above_indices"] = []
            return []

        frame = frame_context.frame
        frame_height, frame_width = frame.shape[:2]
        detections = frame_context.detections
        boxes = detections.boxes
        classes = detections.classes
        scores = detections.scores

        target_class = self.config_obj.target_class.lower()
        conf_threshold = self.config_obj.confidence_threshold

        climbing = []
        fully_above = []
        ever_centers = self._state.get("ever_fully_above_centers") or set()
        ever_centers = set(ever_centers)

        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            if not (isinstance(cls, str) and cls.lower() == target_class and score >= conf_threshold):
                continue
            if not box or len(box) < 4:
                continue

            fully = is_box_fully_above_wall(
                box, self.config_obj.zone_coordinates, frame_width, frame_height
            )
            climb = is_box_climbing(
                box, self.config_obj.zone_coordinates, frame_width, frame_height
            )

            if fully:
                fully_above.append(i)
                # Remember this person's center so we keep them red (once red, stay red)
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                max_b = max(x1, y1, x2, y2)
                if max_b <= 1.0 and frame_width > 1 and frame_height > 1:
                    x1, x2 = x1 * frame_width, x2 * frame_width
                    y1, y2 = y1 * frame_height, y2 * frame_height
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                ever_centers.add((round(cx), round(cy)))
            elif climb:
                climbing.append(i)

        # Cap memory: keep at most 100 centers so set doesn't grow forever
        if len(ever_centers) > 100:
            ever_centers = set()

        # Also mark as red if this detection's center was ever fully above (same person, next frames)
        red_indices = list(fully_above)
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

        self._state["ever_fully_above_centers"] = ever_centers
        self._state["climbing_indices"] = climbing
        self._state["fully_above_indices"] = fully_above
        # For pipeline: we need "who gets red" = red_indices (fully above + ever fully above)
        self._state["red_indices"] = red_indices

        # Emit event when someone is fully above (with cooldown)
        events = []
        if fully_above:
            now = frame_context.timestamp
            last_alert = self._state.get("last_alert_time")
            if last_alert is None or not isinstance(last_alert, datetime) or \
               (now - last_alert).total_seconds() >= self.config_obj.alert_cooldown_seconds:
                self._state["last_alert_time"] = now
                label = self._generate_label(len(fully_above))
                events.append(
                    ScenarioEvent(
                        event_type="wall_climb_detection",
                        label=label,
                        confidence=1.0,
                        metadata={
                            "target_class": self.config_obj.target_class,
                            "fully_above_count": len(fully_above),
                            "alert_cooldown_seconds": self.config_obj.alert_cooldown_seconds,
                        },
                        detection_indices=fully_above,
                        timestamp=frame_context.timestamp,
                        frame_index=frame_context.frame_index,
                    )
                )

        return events

    def _generate_label(self, count: int) -> str:
        if self.config_obj.custom_label and isinstance(self.config_obj.custom_label, str) and self.config_obj.custom_label.strip():
            return self.config_obj.custom_label.strip()
        if count == 1:
            return "Person fully above wall"
        return f"{count} persons fully above wall"

    def reset(self) -> None:
        self._state["last_alert_time"] = None
        self._state["climbing_indices"] = []
        self._state["fully_above_indices"] = []
        self._state["red_indices"] = []
        self._state["ever_fully_above_centers"] = set()
