"""
Class presence configuration
-----------------------------

Parses target class(es), match_mode (any/all), confidence, cooldown. No zone.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict, List


class ClassPresenceConfig:
    """Configuration for class presence scenario."""
    
    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]):
        """
        Initialize configuration from config dict and task.
        
        Args:
            config: Configuration dictionary from rule/task
            task: Full task dictionary (for zone info - not used for this scenario)
        """
        # Parse target class(es) - can be single class or list
        classes = config.get("classes") or config.get("class")
        if classes:
            if isinstance(classes, str):
                self.target_classes = [classes.strip().lower()]
            elif isinstance(classes, list):
                self.target_classes = [c.strip().lower() for c in classes if isinstance(c, str) and c.strip()]
            else:
                self.target_classes = []
        else:
            self.target_classes = []
        
        # Match mode: "any" (default) or "all"
        # "any" = trigger if any of the target classes are detected
        # "all" = trigger only if all target classes are detected
        self.match_mode = str(config.get("match_mode", "any")).lower()
        if self.match_mode not in ["any", "all"]:
            self.match_mode = "any"
        
        # Custom label for alerts
        self.custom_label = config.get("label")
        
        # Confidence threshold for detections
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        
        # Alert cooldown to prevent rapid-fire alerts
        self.alert_cooldown_seconds = float(config.get("alert_cooldown_seconds", 10.0))
        
        # Note: This scenario is NOT zone-based
        # It detects objects across the entire frame
        self.zone_applied = False
