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
        """
        Initialize configuration from config dict and task.
        
        Args:
            config: Configuration dictionary from rule/task
            task: Full task dictionary (for zone info)
        """
        # Default to "box" if no class specified
        self.target_class = str(config.get("class") or "box").strip().lower()
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
        
        # Tracker configuration (optimized for conveyor belt scenarios)
        # 
        # Key settings for stable track ID assignment:
        # - max_age: How long to keep a track alive when not seen (30 frames = ~1 second at 30fps)
        # - min_hits: How many consecutive detections before track is confirmed (1 = immediate)
        # - iou_threshold: Minimum overlap for matching (0.15 = lower threshold for fast-moving boxes)
        # - score_threshold: Minimum detection confidence (0.5 = balanced)
        #
        self.tracker_config = config.get("tracker_config", {})
        self.max_age = self.tracker_config.get("max_age", 30)
        self.min_hits = self.tracker_config.get("min_hits", 1)  # Lower for boxes (immediate confirmation)
        self.iou_threshold = self.tracker_config.get("iou_threshold", 0.15)  # Lower for fast-moving boxes (was 0.3)
        self.score_threshold = self.tracker_config.get("score_threshold", 0.5)
