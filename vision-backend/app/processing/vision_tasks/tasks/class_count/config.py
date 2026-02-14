"""
Class count configuration
-------------------------

Parses target class, zone (line only), direction, tracker settings. Used by class_count scenario.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict, List, Optional


class ClassCountConfig:
    """Configuration for class count scenario."""
    
    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]):
        """
        Initialize configuration from config dict and task.
        
        Args:
            config: Configuration dictionary from rule/task
            task: Full task dictionary (for zone info)
        """
        self.target_class = str(config.get("class") or "").strip().lower()
        self.custom_label = config.get("label")
        
        # Zone configuration (from config or task)
        # Only supports LINE zones for crossing detection
        zone = config.get("zone") or task.get("zone")
        self.zone_type: Optional[str] = None
        self.zone_coordinates: Optional[List[List[float]]] = None
        self.zone_direction: str = "both"
        self.zone_applied = False
        
        if zone and isinstance(zone, dict):
            zone_type = zone.get("type", "").lower()
            coords = zone.get("coordinates")
            
            # Only support line zones (2 points for crossing detection)
            if zone_type == "line" and isinstance(coords, list) and len(coords) == 2:
                self.zone_type = "line"
                self.zone_coordinates = coords
                self.zone_direction = zone.get("direction", "both")
                self.zone_applied = True
        
        # Tracker configuration (for line-based counting)
        # Lower min_hits for faster confirmation (objects may move quickly)
        self.tracker_config = config.get("tracker_config", {})
        self.max_age = self.tracker_config.get("max_age", 30)
        self.min_hits = self.tracker_config.get("min_hits", 1)  # Lower for faster confirmation (was 3)
        self.iou_threshold = self.tracker_config.get("iou_threshold", 0.3)
        self.score_threshold = self.tracker_config.get("score_threshold", 0.5)
