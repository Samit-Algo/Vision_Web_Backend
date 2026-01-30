"""
Box Count Configuration
-----------------------

Handles configuration parsing for box count scenario.
Same as class_count but defaults to "box" class.
"""

from typing import Optional, Dict, Any, List


class BoxCountConfig:
    """Configuration for box count scenario."""

    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]):
        self.target_class = str(config.get("class") or "box").strip().lower()
        self.custom_label = config.get("label")
        zone = config.get("zone") or task.get("zone")
        self.zone_type: Optional[str] = None
        self.zone_coordinates: Optional[List[List[float]]] = None
        self.zone_direction: str = "both"
        self.zone_applied = False
        if zone and isinstance(zone, dict):
            zone_type = zone.get("type", "").lower()
            coords = zone.get("coordinates")
            if zone_type == "line" and isinstance(coords, list) and len(coords) == 2:
                self.zone_type = "line"
                self.zone_coordinates = coords
                self.zone_direction = zone.get("direction", "both")
                self.zone_applied = True
        tracker_config = config.get("tracker_config", {})
        self.max_age = tracker_config.get("max_age", 30)
        self.min_hits = tracker_config.get("min_hits", 1)
        self.iou_threshold = tracker_config.get("iou_threshold", 0.15)
        self.score_threshold = tracker_config.get("score_threshold", 0.5)
