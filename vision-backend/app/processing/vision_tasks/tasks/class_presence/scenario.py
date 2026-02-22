"""
Class Presence Scenario
------------------------
Detects if target class(es) are present anywhere in the frame (no zone). Match mode: any or all.
Alerts with cooldown when configured classes detected.

Code layout:
  - ClassPresenceScenario: __init__, process (find_matching_detections → state → event if cooldown ok), find_matching_detections, generate_label, reset.
"""

# -------- Imports --------
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from .config import ClassPresenceConfig
from .detector import (
    normalize_classes,
    find_matched_classes,
    find_matched_indices,
    generate_label,
)


# ========== Scenario: Class presence (target classes in frame, no zone) ==========

@register_scenario("class_presence")
class ClassPresenceScenario(BaseScenario):
    """
    Detects if specified class(es) are present in the frame.
    
    This scenario:
    - Filters all detections to only show user-specified class(es)
    - Detects objects across the entire frame (NOT zone-based)
    - Shows bounding boxes only for the filtered class(es)
    - Triggers alerts when specified class(es) are detected
    """

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = ClassPresenceConfig(config, pipeline_context.task)
        self._state["last_alert_time"] = None
        self._state["class_detected"] = False
        self._state["detected_classes"] = []

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process a single frame.
        
        Filters detections by user-specified class(es) and triggers alert if detected.
        """
        if not self.config_obj.target_classes:
            self._state["class_detected"] = False
            self._state["detected_classes"] = []
            return []
        matched_indices, matched_classes, matched_scores = self.find_matching_detections(
            frame_context
        )
        if matched_indices:
            self._state["class_detected"] = True
            self._state["detected_classes"] = matched_classes
        else:
            self._state["class_detected"] = False
            self._state["detected_classes"] = []
        if not matched_indices:
            return []
        # --- Emit event if not in cooldown ---
        now = frame_context.timestamp
        last_alert_time = self._state.get("last_alert_time")
        if last_alert_time and isinstance(last_alert_time, datetime):
            if (now - last_alert_time).total_seconds() < self.config_obj.alert_cooldown_seconds:
                # Still in cooldown, but return event with detection_indices for visualization
                # (bounding boxes should still be shown)
                return [
                    ScenarioEvent(
                        event_type="class_presence",
                        label=self.generate_label(matched_classes),
                        confidence=max(matched_scores) if matched_scores else 0.0,
                        metadata={
                            "target_classes": self.config_obj.target_classes,
                            "detected_classes": matched_classes,
                            "detection_count": len(matched_indices),
                            "match_mode": self.config_obj.match_mode,
                            "zone_applied": False,
                            "alert_cooldown_seconds": self.config_obj.alert_cooldown_seconds,
                            "in_cooldown": True,
                        },
                        detection_indices=matched_indices,
                        timestamp=frame_context.timestamp,
                        frame_index=frame_context.frame_index,
                    )
                ]

        # Trigger alert (not in cooldown)
        self._state["last_alert_time"] = now

        # Calculate confidence metrics
        avg_confidence = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0
        max_confidence = max(matched_scores) if matched_scores else 0.0

        return [
            ScenarioEvent(
                event_type="class_presence",
                label=self.generate_label(matched_classes),
                confidence=max_confidence,
                metadata={
                    "target_classes": self.config_obj.target_classes,
                    "detected_classes": matched_classes,
                    "detection_count": len(matched_indices),
                    "match_mode": self.config_obj.match_mode,
                    "avg_confidence": avg_confidence,
                    "max_confidence": max_confidence,
                    "zone_applied": False,
                    "alert_cooldown_seconds": self.config_obj.alert_cooldown_seconds,
                    "in_cooldown": False,
                },
                detection_indices=matched_indices,
                timestamp=frame_context.timestamp,
                frame_index=frame_context.frame_index,
            )
        ]

    def find_matching_detections(
        self, frame_context: ScenarioFrameContext
    ) -> Tuple[List[int], List[str], List[float]]:
        """
        Find detections matching the target class(es).
        
        Returns:
            Tuple of (matched_indices, matched_classes, matched_scores)
        """
        detections = frame_context.detections
        boxes = detections.boxes
        classes = detections.classes
        scores = detections.scores

        matched_indices = []
        matched_classes = []
        matched_scores = []

        # Normalize target classes
        target_classes_normalized = normalize_classes(self.config_obj.target_classes)
        
        # Get all detected classes (normalized)
        detected_classes_normalized = normalize_classes(classes)

        # Check if match condition is met based on match_mode
        matched_class_names, matched_now = find_matched_classes(
            detected_classes_normalized,
            target_classes_normalized,
            self.config_obj.match_mode
        )

        # If match condition is met, find all detection indices for matched classes
        if matched_now:
            for idx, (detected_class, score) in enumerate(zip(classes, scores)):
                if isinstance(detected_class, str):
                    detected_class_normalized = detected_class.lower()
                    # Check if this detection matches any of the matched classes
                    if detected_class_normalized in matched_class_names:
                        # Apply confidence threshold
                        if score >= self.config_obj.confidence_threshold:
                            matched_indices.append(idx)
                            matched_classes.append(detected_class)
                            matched_scores.append(float(score))

        return matched_indices, matched_classes, matched_scores

    def generate_label(self, matched_classes: List[str]) -> str:
        """Generate event label."""
        return generate_label(
            matched_classes,
            self.config_obj.target_classes,
            self.config_obj.match_mode,
            self.config_obj.custom_label
        )

    def reset(self) -> None:
        """Reset scenario state."""
        self._state["last_alert_time"] = None
        self._state["class_detected"] = False
        self._state["detected_classes"] = []
