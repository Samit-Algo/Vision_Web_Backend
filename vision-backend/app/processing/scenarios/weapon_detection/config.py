"""
Weapon Detection Configuration
------------------------------

Handles configuration parsing and validation for weapon detection scenario.
"""

from typing import Dict, Any


class WeaponDetectionConfig:
    """Configuration for weapon detection scenario."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize configuration from config dict.
        
        Args:
            config: Configuration dictionary from rule/task
        """
        self.buffer_size = int(config.get("buffer_size", 5))
        self.arm_angle_threshold = float(config.get("arm_angle_threshold", 45.0))
        self.vlm_confidence_threshold = float(config.get("vlm_confidence_threshold", 0.7))
        self.temporal_consistency_frames = int(config.get("temporal_consistency_frames", 4))
        self.vlm_throttle_seconds = float(config.get("vlm_throttle_seconds", 2.0))
        self.vlm_frames_dir = config.get("vlm_frames_dir", "vlm_frames")
