"""
Fire Detection Scenario
------------------------
Detects fire/flame/smoke anywhere in frame (no zone). Alert when fire confirmed for confirm_frames
and not in alert_cooldown. State: consecutive_fire_frames, fire_detected, last_alert_time.

Code layout:
  - FireDetectionScenario: __init__ (state), process (find_fire → update state → emit event if alert),
    find_fire_detections, generate_label, reset.
"""

# -------- Imports --------
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioEvent,
    ScenarioFrameContext,
)
from app.processing.vision_tasks.task_lookup import register_scenario

from .config import FireDetectionConfig

# -----------------------------------------------------------------------------
# Scenario
# -----------------------------------------------------------------------------


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
        # --- Find fire-like classes (fire, flame, smoke) above confidence ---
        fire_detections = self.find_fire_detections(frame_context)
        if not fire_detections:
            if self._state["consecutive_fire_frames"] > 0:
                self._state["consecutive_fire_frames"] -= 1
            if self._state["consecutive_fire_frames"] == 0:
                self._state["fire_detected"] = False
            return []
        # --- Update state: consecutive_fire_frames, fire_detected ---
        self._state["consecutive_fire_frames"] += 1
        self._state["total_fire_detections"] += 1
        matched_indices, matched_classes, matched_scores = fire_detections
        avg_confidence = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0
        max_confidence = max(matched_scores) if matched_scores else 0.0
        self._state["fire_detected"] = True
        now = frame_context.timestamp
        is_confirmed = self._state["consecutive_fire_frames"] >= self.config_obj.confirm_frames
        # --- Emit alert only if confirmed and not in cooldown ---
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
            label = self.generate_label(len(matched_indices), matched_classes)
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
                    "alert_cooldown_seconds": self.config_obj.alert_cooldown_seconds,
                },
                detection_indices=matched_indices,
                timestamp=frame_context.timestamp,
                frame_index=frame_context.frame_index,
            )
        ]

    def find_fire_detections(
        self, frame_context: ScenarioFrameContext
    ) -> Optional[Tuple[List[int], List[str], List[float]]]:
        """Return (indices, classes, scores) for detections matching target_classes (fire/flame/smoke) above confidence, or None."""
        detections = frame_context.detections
        classes = detections.classes
        scores = detections.scores
        matched_indices = []
        matched_classes = []
        matched_scores = []
        target_classes = [c.lower() for c in self.config_obj.target_classes]
        confidence_threshold = self.config_obj.confidence_threshold
        for i, (cls, score) in enumerate(zip(classes, scores)):
            if isinstance(cls, str):
                cls_lower = cls.lower()
                is_fire_class = any(
                    target in cls_lower or cls_lower in target for target in target_classes
                )
                if is_fire_class and score >= confidence_threshold:
                    matched_indices.append(i)
                    matched_classes.append(cls)
                    matched_scores.append(float(score))
        if matched_indices:
            return matched_indices, matched_classes, matched_scores
        return None

    def generate_label(self, count: int, classes: List[str]) -> str:
        """Build alert label: custom if set, else fire/smoke text by detected classes."""
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
        """Clear all fire-detection state."""
        self._state["last_alert_time"] = None
        self._state["fire_detected"] = False
        self._state["consecutive_fire_frames"] = 0
        self._state["total_fire_detections"] = 0
        self._state["fire_detection_history"] = []
        self._state["detection_counts"] = defaultdict(int)
