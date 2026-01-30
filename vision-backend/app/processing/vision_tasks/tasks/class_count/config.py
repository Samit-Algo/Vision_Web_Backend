"""
Class Count Configuration
--------------------------

Handles configuration parsing for class count scenario.
"""

from typing import Optional, Dict, Any, List


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
        
        # Zone configuration (from task)
        zone = task.get("zone")
        self.zone_coordinates: Optional[List[List[float]]] = None
        self.zone_applied = False
        
        if zone and isinstance(zone, dict):
            zone_type = zone.get("type", "").lower()
            if zone_type == "polygon":
                coords = zone.get("coordinates")
                if isinstance(coords, list) and len(coords) >= 3:
                    self.zone_coordinates = coords
                    self.zone_applied = True
