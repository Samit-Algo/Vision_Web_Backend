"""
Fall detection configuration
-----------------------------

Parses target class (person), pose keypoint thresholds, hip drop, lying angle, confirm_frames, cooldown.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict


class FallDetectionConfig:
    """Config for fall detection: person class, keypoint thresholds, confirm frames, cooldown."""

    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]) -> None:
        self.target_class = str(config.get("class") or "person").strip().lower()
        self.custom_label = config.get("label")
        self.alert_cooldown_seconds = float(config.get("alert_cooldown_seconds", 10))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        self.kp_confidence_threshold = float(config.get("kp_confidence_threshold", 0.3))
        self.lying_angle_threshold = float(config.get("lying_angle_threshold", 50))
        self.hip_drop_threshold = float(config.get("hip_drop_threshold", 15))
        self.height_drop_ratio = float(config.get("height_drop_ratio", 0.25))
        self.min_height_for_standing = float(config.get("min_height_for_standing", 300))
        self.confirm_frames = int(config.get("confirm_frames", 3))
        self.enabled = config.get("enabled", True)
