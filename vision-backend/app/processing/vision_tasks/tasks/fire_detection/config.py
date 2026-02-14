"""
Fire detection configuration
-----------------------------

Parses target classes (fire, flame, smoke), confidence, confirm_frames, and alert cooldown.
No zone: fire anywhere in the frame is considered. Alert is driven only by user settings.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict, List


class FireDetectionConfig:
    """Config for fire detection: target classes and alert thresholds. Zone is not used."""

    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]) -> None:
        self.target_classes = self.parse_target_classes(config)
        self.custom_label = config.get("label")
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        self.alert_cooldown_seconds = float(config.get("alert_cooldown_seconds", 10.0))
        self.confirm_frames = int(config.get("confirm_frames", 2))

    def parse_target_classes(self, config: Dict[str, Any]) -> List[str]:
        """Parse classes to detect; default fire, flame, smoke."""
        default_classes = ["fire", "flame", "smoke"]
        classes = config.get("classes") or config.get("class")
        if not classes:
            return default_classes
        if isinstance(classes, str):
            return [classes.strip().lower()]
        if isinstance(classes, list):
            return [c.strip().lower() for c in classes if isinstance(c, str)]
        return default_classes
