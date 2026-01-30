"""
Restricted Zone Configuration
-----------------------------

Handles configuration parsing for restricted zone scenario.
"""

from typing import Optional, Dict, Any, List


class RestrictedZoneConfig:
    """Configuration for restricted zone scenario."""

    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]):
        self.target_class = str(config.get("class") or "").strip().lower()
        self.custom_label = config.get("label")
        self.alert_cooldown_seconds = float(config.get("alert_cooldown_seconds", 10))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.7))
        zone = config.get("zone") or task.get("zone")
        self.zone_coordinates: Optional[List[List[float]]] = None
        if zone and isinstance(zone, dict):
            zone_type = zone.get("type", "").lower()
            coords = zone.get("coordinates")
            if zone_type == "polygon" and isinstance(coords, list) and len(coords) >= 3:
                self.zone_coordinates = coords
