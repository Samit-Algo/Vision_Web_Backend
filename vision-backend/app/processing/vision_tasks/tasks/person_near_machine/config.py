"""
Person Near Machine Configuration
----------------------------------

Uses polygon zone (same as restricted_zone): zone_coordinates from zone.type "polygon".
Stores exact user-drawn polygon, not ROI/rectangle.
"""

from typing import Any, Dict, List, Optional


# ============================================================================
# CONFIG
# ============================================================================

class PersonNearMachineConfig:
    """Configuration for person near machine monitoring. Uses polygon zone (zone_coordinates)."""

    def __init__(self, config: Dict[str, Any], task: Optional[Dict[str, Any]] = None):
        """
        Expected structure (same zone as restricted_zone):
        - rule.zone = { "type": "polygon", "coordinates": [[x,y], [x,y], ...] }  (at least 3 points)
        - rule.config = { absence_threshold_minutes, grace_time_seconds, min_presence_seconds, ... }
        """
        task = task or {}
        zone = config.get("zone") or task.get("zone")
        self.zone_coordinates: Optional[List[List[float]]] = None
        if zone and isinstance(zone, dict) and str(zone.get("type", "")).lower() == "polygon":
            coords = zone.get("coordinates")
            if isinstance(coords, list) and len(coords) >= 3:
                self.zone_coordinates = [[float(p[0]), float(p[1])] for p in coords]

        config_params = config.get("config", {})

        # absence_threshold_minutes must be provided by the user
        raw_absence = config_params.get("absence_threshold_minutes")
        if raw_absence is None:
            raise ValueError(
                "absence_threshold_minutes is required. Please provide the unattended threshold (in minutes) during agent creation."
            )
        self.absence_threshold_minutes = float(raw_absence)
        self.grace_time_seconds = float(config_params.get("grace_time_seconds", 10.0))
        self.min_presence_seconds = float(config_params.get("min_presence_seconds", 5.0))

        self.confidence_threshold = float(config_params.get("confidence_threshold", 0.5))
        self.tracker_max_age = int(config_params.get("tracker_max_age", 30))
        self.tracker_min_hits = int(config_params.get("tracker_min_hits", 3))
        self.tracker_iou_threshold = float(config_params.get("tracker_iou_threshold", 0.3))
        self.notify_on_absence = bool(config_params.get("notify_on_absence", True))
        self.alert_cooldown_minutes = float(config_params.get("alert_cooldown_minutes", 30.0))
        self.emit_state_transitions = bool(config_params.get("emit_state_transitions", False))
        self.emit_periodic_updates = bool(config_params.get("emit_periodic_updates", False))

        # Process 5 frames per second (5 FPS)
        self.fps = int(config_params.get("fps", 5))
        if self.fps < 1:
            self.fps = 5

        if self.absence_threshold_minutes <= 0.0:
            raise ValueError("absence_threshold_minutes must be > 0")
        if self.grace_time_seconds < 0.0:
            raise ValueError("grace_time_seconds must be >= 0")
        if self.min_presence_seconds < 0.0:
            raise ValueError("min_presence_seconds must be >= 0")
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0]")
        if self.alert_cooldown_minutes < 0.0:
            raise ValueError("alert_cooldown_minutes must be >= 0")
