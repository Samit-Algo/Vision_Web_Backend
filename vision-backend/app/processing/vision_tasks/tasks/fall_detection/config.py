"""
Fall detection configuration
-----------------------------

Parses target class (person), pose keypoint thresholds, hip drop, lying angle,
aspect ratio (horizontal body), head-hip alignment, recovery timeout, cooldown.
State machine: NORMAL → FALL_SUSPECTED → (RECOVERED or TIMEOUT) → NORMAL or CONFIRMED_FALL.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict


class FallDetectionConfig:
    """Config for fall detection: person class, keypoint thresholds, recovery timeout, cooldown."""

    def __init__(self, config: Dict[str, Any], task: Dict[str, Any]) -> None:
        self.target_class = str(config.get("class") or "person").strip().lower()
        self.custom_label = config.get("label")
        self.alert_cooldown_seconds = float(config.get("alert_cooldown_seconds", 10))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        self.kp_confidence_threshold = float(config.get("kp_confidence_threshold", 0.25))

        # --- Recovery & confirmation (state machine) ---
        # Seconds to wait for recovery before confirming fall. 3 = fast alert, 5 = fewer false alarms.
        self.recovery_timeout_seconds = float(config.get("recovery_timeout_seconds", 3.0))
        # Height ratio: current_height > fall_height * this => recovered.
        self.recovery_height_ratio = float(config.get("recovery_height_ratio", 1.3))
        # Torso angle (deg) below this => vertical / recovered.
        self.recovery_torso_angle_max = float(config.get("recovery_torso_angle_max", 35.0))
        # Hip/head must move up by this ratio of fall height to confirm recovery
        self.recovery_hip_lift_ratio = float(config.get("recovery_hip_lift_ratio", 0.15))

        # --- Fall SUSPECTED triggers (multiple independent signals) ---
        # Absolute pixel drop: hip/head moved down by this many pixels => suspicious (works at any distance)
        self.hip_drop_pixels_threshold = float(config.get("hip_drop_pixels_threshold", 30.0))
        self.head_drop_pixels_threshold = float(config.get("head_drop_pixels_threshold", 30.0))
        # Relative drop: hip/head moved down by this ratio of person height => suspicious
        self.hip_drop_ratio_threshold = float(config.get("hip_drop_ratio_threshold", 0.15))
        self.head_drop_ratio_threshold = float(config.get("head_drop_ratio_threshold", 0.15))
        # Velocity: downward velocity (heights per second) above this => suspicious
        self.sudden_drop_velocity_threshold = float(config.get("sudden_drop_velocity_threshold", 1.2))
        # Height collapse: (prev_height - current_height) / prev_height > this => suspicious
        self.height_collapse_ratio = float(config.get("height_collapse_ratio", 0.20))
        # Torso horizontal: angle from vertical (deg) above this => suspicious
        self.torso_horizontal_angle_threshold = float(config.get("torso_horizontal_angle_threshold", 45.0))
        # Aspect ratio: width/height > this => person is horizontal (lying down)
        self.aspect_ratio_threshold = float(config.get("aspect_ratio_threshold", 1.0))
        # Head-hip vertical separation: when lying, head and hip are at similar vertical level
        self.head_hip_vertical_ratio_max = float(config.get("head_hip_vertical_ratio_max", 0.5))

        # --- VLM Configuration ---
        # Enable/disable VLM confirmation (if False, uses timeout-based confirmation)
        self.vlm_enabled = config.get("vlm_enabled", True)
        # Minimum confidence from VLM to confirm fall (0.0-1.0)
        self.vlm_confidence_threshold = float(config.get("vlm_confidence_threshold", 0.7))
        # Minimum seconds between VLM calls for the same person (throttling)
        self.vlm_throttle_seconds = float(config.get("vlm_throttle_seconds", 5.0))
        # Directory to save frames sent to VLM (for debugging)
        self.vlm_frames_dir = str(config.get("vlm_frames_dir", "./vlm_frames/fall_detection"))
        # Size of frame buffer for VLM (should be >= 5 for 5-frame VLM call)
        self.vlm_buffer_size = int(config.get("vlm_buffer_size", 20))

        # Legacy / compatibility (used as fallbacks in analysis)
        self.lying_angle_threshold = float(config.get("lying_angle_threshold", 38))
        self.hip_drop_threshold = float(config.get("hip_drop_threshold", 12))
        self.height_drop_ratio = float(config.get("height_drop_ratio", 0.22))
        self.min_height_for_standing = float(config.get("min_height_for_standing", 180))
        self.aspect_ratio_threshold = float(config.get("aspect_ratio_threshold", 0.95))
        self.head_hip_vertical_ratio_max = float(config.get("head_hip_vertical_ratio_max", 0.45))
        self.confirm_frames = int(config.get("confirm_frames", 2))
        self.enabled = config.get("enabled", True)
