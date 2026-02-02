"""
Sleep Detection Configuration
-----------------------------

Reads settings from the rule config (e.g. from UI).
All values have defaults so it works out of the box.
"""

from typing import Dict, Any


class SleepDetectionConfig:
    """Configuration for the sleep detection scenario."""

    def __init__(self, config: Dict[str, Any]):
        """
        Read config from the rule dictionary.
        If a key is missing, we use a sensible default.
        """
        # How many frames to keep in the buffer (we need at least 3 for before/suspicious/after)
        self.buffer_size = int(config.get("buffer_size", 5))

        # How many frames in a row must look "possibly sleeping" before we call the VLM
        self.temporal_consistency_frames = int(config.get("temporal_consistency_frames", 4))

        # VLM: only emit alert if VLM confidence is at least this (0.0–1.0)
        self.vlm_confidence_threshold = float(config.get("vlm_confidence_threshold", 0.7))

        # Don't call VLM for the same person more than once per this many seconds
        self.vlm_throttle_seconds = float(config.get("vlm_throttle_seconds", 2.0))

        # Wait this many seconds of "possibly sleeping + no movement" before sending frames to VLM (cost-efficient)
        self.vlm_trigger_still_seconds = float(config.get("vlm_trigger_still_seconds", 5.0))

        # Folder where we save the 3 frames before sending to VLM (optional, for debugging)
        self.vlm_frames_dir = config.get("vlm_frames_dir", "sleep_vlm_frames")

        # Only use keypoints with confidence >= this (0 = use all)
        self.kp_confidence_threshold = float(config.get("kp_confidence_threshold", 0.0))

        # --- Pose rules for "possibly sleeping" ---
        # Lying: torso (shoulder–hip) angle within this many degrees of horizontal (0 = flat)
        self.torso_angle_lying_deg = float(config.get("torso_angle_lying_deg", 35.0))

        # Standing sleep: head is "down" if nose is this many degrees below shoulder line
        # Lower value (e.g. 25) catches chin-down / head-forward posture more easily
        self.head_down_angle_deg = float(config.get("head_down_angle_deg", 25.0))

        # Alternative: trigger "head down" if nose is at least this many pixels below shoulder (0 = disable)
        self.min_nose_below_shoulder_px = float(config.get("min_nose_below_shoulder_px", 10.0))

        # For head-down standing sleep: require this fraction of frames to show head down (0.5 = 2 of 4)
        self.head_down_majority_ratio = float(config.get("head_down_majority_ratio", 0.5))

        # Motion: if keypoints move less than this (pixels) on average, consider "still"
        self.motion_threshold_px = float(config.get("motion_threshold_px", 25.0))
