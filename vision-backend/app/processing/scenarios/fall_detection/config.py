"""
Fall Detection Configuration
----------------------------

Handles configuration parsing for fall detection scenario.
"""

from typing import Optional, Dict, Any


class FallDetectionConfig:
    """Configuration for fall detection scenario."""
    
    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]):
        """
        Initialize configuration from config dict and task.
        
        Args:
            config: Configuration dictionary from rule/task
            task: Full task dictionary
        """
        # Target class to monitor (default: person)
        self.target_class = str(config.get("class") or "person").strip().lower()
        
        # Custom label for alerts
        self.custom_label = config.get("label")
        
        # Alert cooldown (seconds between alerts)
        self.alert_cooldown_seconds = float(config.get("alert_cooldown_seconds", 10))
        
        # Confidence threshold for detections
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        
        # Keypoint confidence threshold
        self.kp_confidence_threshold = float(config.get("kp_confidence_threshold", 0.3))
        
        # Fall detection thresholds
        self.lying_angle_threshold = float(config.get("lying_angle_threshold", 50))  # degrees from vertical
        self.hip_drop_threshold = float(config.get("hip_drop_threshold", 15))  # pixels
        self.height_drop_ratio = float(config.get("height_drop_ratio", 0.25))  # 25% height collapse
        self.min_height_for_standing = float(config.get("min_height_for_standing", 300))  # pixels
        
        # Confirmation frames (how many consecutive frames to confirm fall)
        self.confirm_frames = int(config.get("confirm_frames", 3))
        
        # Enable/disable fall detection
        self.enabled = config.get("enabled", True)
