"""
Wall climb configuration
-------------------------

Parses zone polygon (wall boundary), target class, confidence, cooldown, and VLM settings.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict, List, Optional


class WallClimbConfig:
    """Configuration for wall climb detection."""

    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]):
        # Class to detect (e.g. "person")
        self.target_class = str(config.get("class") or "person").strip().lower()
        self.custom_label = config.get("label")
        self.alert_cooldown_seconds = float(config.get("alert_cooldown_seconds", 10))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.3))

        # Zone = polygon drawn by user (wall boundary)
        zone = config.get("zone") or task.get("zone")
        self.zone_coordinates: Optional[List[List[float]]] = None
        if zone and isinstance(zone, dict):
            zone_type = zone.get("type", "").lower()
            coords = zone.get("coordinates")
            if zone_type == "polygon" and isinstance(coords, list) and len(coords) >= 3:
                self.zone_coordinates = coords

        # VLM Configuration
        # How many frames to keep in the buffer (we need at least 3 for before/suspicious/after)
        self.buffer_size = int(config.get("buffer_size", 5))
        
        # VLM: only emit alert if VLM confidence is at least this (0.0–1.0)
        self.vlm_confidence_threshold = float(config.get("vlm_confidence_threshold", 0.7))
        
        # Don't call VLM for the same person more than once per this many seconds
        self.vlm_throttle_seconds = float(config.get("vlm_throttle_seconds", 2.0))
        
        # Folder where we save the 3 frames before sending to VLM (optional, for debugging)
        self.vlm_frames_dir = config.get("vlm_frames_dir", "wall_climb_vlm_frames")

        # Tracker: keep same track_id when person reappears after occlusion (Kalman + longer max_age)
        tracker_config = config.get("tracker_config") or {}
        self.tracker_max_age = int(tracker_config.get("max_age", 180))  # frames to keep lost track (default 180 ≈ 6s @30fps)
        self.tracker_min_hits = int(tracker_config.get("min_hits", 2))
        self.tracker_iou_threshold = float(tracker_config.get("iou_threshold", 0.3))
        self.tracker_score_threshold = float(tracker_config.get("score_threshold", self.confidence_threshold))
        self.tracker_max_distance = float(tracker_config.get("max_distance_threshold", 180.0))
        self.tracker_max_distance_max = float(tracker_config.get("max_distance_threshold_max", 400.0))
        self.tracker_distance_growth = float(tracker_config.get("distance_growth_per_missed_frame", 10.0))
        self.tracker_use_kalman = bool(tracker_config.get("use_kalman", True))
