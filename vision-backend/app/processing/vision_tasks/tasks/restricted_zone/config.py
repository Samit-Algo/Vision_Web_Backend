"""
Restricted zone configuration
------------------------------

Parses target class, confidence, alert cooldown, polygon zone coordinates, and tracking parameters.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict, List, Optional


class RestrictedZoneConfig:
    """Config for restricted zone: target class, zone polygon, cooldown, tracking."""

    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]) -> None:
        self.target_class = str(config.get("class") or "").strip().lower()
        self.custom_label = config.get("label")
        self.alert_cooldown_seconds = float(config.get("alert_cooldown_seconds", 10))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.4))
        
        # Zone configuration
        zone = config.get("zone") or task.get("zone")
        self.zone_coordinates: Optional[List[List[float]]] = None
        if zone and isinstance(zone, dict):
            zone_type = zone.get("type", "").lower()
            coords = zone.get("coordinates")
            if zone_type == "polygon" and isinstance(coords, list) and len(coords) >= 3:
                self.zone_coordinates = coords
        
        # Duration-based alerting
        self.duration_threshold_seconds = float(config.get("duration_threshold_seconds", 2.0))
        self.duration_alert_interval_seconds = float(config.get("duration_alert_interval_seconds", 1.0))
        
        # Instant trigger: 1 = no anti-flicker wait (alert as soon as person in zone)
        self.stability_frames = int(config.get("stability_frames", 1))
        
        # Tracker: min_hits=1 so new detections get a track immediately (instant zone check)
        tracker_config = config.get("tracker_config") or {}
        self.tracker_max_age = int(tracker_config.get("max_age", 30))
        self.tracker_min_hits = int(tracker_config.get("min_hits", 1))
        self.tracker_iou_threshold = float(tracker_config.get("iou_threshold", 0.3))
        self.tracker_score_threshold = float(tracker_config.get("score_threshold", self.confidence_threshold))
        self.tracker_max_distance = float(tracker_config.get("max_distance_threshold", 150.0))
        self.tracker_max_distance_max = float(tracker_config.get("max_distance_threshold_max", 350.0))
        self.tracker_distance_growth = float(tracker_config.get("distance_growth_per_missed_frame", 8.0))
        self.tracker_use_kalman = bool(tracker_config.get("use_kalman", False))
