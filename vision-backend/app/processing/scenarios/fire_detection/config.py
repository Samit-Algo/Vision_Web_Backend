"""
Fire Detection Configuration
----------------------------

Handles configuration parsing for fire detection scenario.
"""

from typing import Optional, Dict, Any, List


class FireDetectionConfig:
    """Configuration for fire detection scenario."""
    
    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]):
        """
        Initialize configuration from config dict and task.
        
        Args:
            config: Configuration dictionary from rule/task
            task: Full task dictionary (for zone info)
        """
        # Target classes for fire detection
        # The fine-tuned model detects: fire, smoke (possibly)
        self.target_classes = self._parse_target_classes(config)
        
        # Custom label for alerts
        self.custom_label = config.get("label")
        
        # Confidence threshold for fire detection
        # Higher threshold reduces false positives
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        
        # Alert cooldown (seconds) to prevent spam
        self.alert_cooldown_seconds = float(config.get("alert_cooldown_seconds", 10.0))
        
        # Minimum consecutive frames with fire to confirm detection
        # Helps reduce false positives from flickering lights, etc.
        self.confirm_frames = int(config.get("confirm_frames", 2))
        
        # Zone configuration (optional - for monitoring specific areas)
        zone = config.get("zone") or task.get("zone")
        self.zone_type: Optional[str] = None
        self.zone_coordinates: Optional[List[List[float]]] = None
        self.zone_applied = False
        
        if zone and isinstance(zone, dict):
            zone_type = zone.get("type", "").lower()
            coords = zone.get("coordinates")
            
            # Support polygon zones for restricting detection area
            if zone_type == "polygon" and isinstance(coords, list) and len(coords) >= 3:
                self.zone_type = "polygon"
                self.zone_coordinates = coords
                self.zone_applied = True
    
    def _parse_target_classes(self, config: Dict[str, Any]) -> List[str]:
        """
        Parse target classes from config.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of target class names to detect
        """
        # Default classes for fire detection
        default_classes = ["fire", "flame", "smoke"]
        
        # Get from config if specified
        classes = config.get("classes") or config.get("class")
        
        if classes:
            if isinstance(classes, str):
                return [classes.strip().lower()]
            elif isinstance(classes, list):
                return [c.strip().lower() for c in classes if isinstance(c, str)]
        
        return default_classes
