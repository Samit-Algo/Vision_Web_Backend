"""
Fire Detection Configuration
-----------------------------

Handles configuration parsing for fire detection scenario.
"""

from typing import Optional, Dict, Any, List


class FireDetectionConfig:
    """Configuration for fire detection scenario."""

    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]):
        self.target_classes = self._parse_target_classes(config)
        self.custom_label = config.get("label")
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        self.alert_cooldown_seconds = float(config.get("alert_cooldown_seconds", 10.0))
        self.confirm_frames = int(config.get("confirm_frames", 2))
        zone = config.get("zone") or task.get("zone")
        self.zone_type: Optional[str] = None
        self.zone_coordinates: Optional[List[List[float]]] = None
        self.zone_applied = False
        if zone and isinstance(zone, dict):
            zone_type = zone.get("type", "").lower()
            coords = zone.get("coordinates")
            if zone_type == "polygon" and isinstance(coords, list) and len(coords) >= 3:
                self.zone_type = "polygon"
                self.zone_coordinates = coords
                self.zone_applied = True

    def _parse_target_classes(self, config: Dict[str, Any]) -> List[str]:
        default_classes = ["fire", "flame", "smoke"]
        classes = config.get("classes") or config.get("class")
        if classes:
            if isinstance(classes, str):
                return [classes.strip().lower()]
            if isinstance(classes, list):
                return [c.strip().lower() for c in classes if isinstance(c, str)]
        return default_classes
