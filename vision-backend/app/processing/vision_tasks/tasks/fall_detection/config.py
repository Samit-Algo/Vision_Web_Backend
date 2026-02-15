"""
Fall detection configuration
-----------------------------

Parses target class (person), pose keypoint thresholds, hip drop, lying angle,
aspect ratio (horizontal body), head-hip alignment, confirm_frames, cooldown.
Tuned so that "person fall and laydown" (supine/horizontal) is reliably detected.
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
        self.kp_confidence_threshold = float(config.get("kp_confidence_threshold", 0.25))
        # Torso angle from vertical (deg): standing ~0, lying ~90. Lower = catch more laydown poses.
        self.lying_angle_threshold = float(config.get("lying_angle_threshold", 38))
        self.hip_drop_threshold = float(config.get("hip_drop_threshold", 12))
        self.height_drop_ratio = float(config.get("height_drop_ratio", 0.22))
        # Min keypoint bbox height (px) to consider "standing"; below = collapsed/lying.
        self.min_height_for_standing = float(config.get("min_height_for_standing", 180))
        # Keypoint bbox: width/height > this => person horizontal (laydown). 1.0 = wider than tall.
        self.aspect_ratio_threshold = float(config.get("aspect_ratio_threshold", 0.95))
        # Head-hip vertical separation / bbox_height below this => body horizontal (laydown).
        self.head_hip_vertical_ratio_max = float(config.get("head_hip_vertical_ratio_max", 0.45))
        # How many consecutive frames with "lying" to confirm fall (2 = faster, less missed).
        self.confirm_frames = int(config.get("confirm_frames", 2))
        self.enabled = config.get("enabled", True)
