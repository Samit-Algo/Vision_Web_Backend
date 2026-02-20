"""
Loom Machine State Configuration
--------------------------------

Handles configuration parsing and validation for loom machine state scenario.
Supports idle-alert: notify when a machine is idle (no pixel movement) for longer
than idle_threshold_minutes. Uses frames_per_minute and analysis_window_minutes
for sampling and window-based idle detection.
"""

from typing import Any, Dict, List


# ============================================================================
# CONFIG
# ============================================================================

class LoomMachineStateConfig:
    """Configuration for loom machine state scenario."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize configuration from config dict.

        Expected structure:
        {
            "looms": [
                {
                    "loom_id": "loom-01",
                    "motion_roi": [100, 200, 300, 400],
                    "name": "Loom Machine #1",
                    "position": "Line A, Position 1",  # Optional
                    "line_number": "A"  # Optional
                }
            ],
            "config": {
                "motion_threshold": 0.02,
                "optical_flow_threshold": 0.5,
                "temporal_consistency_seconds": 2.0,
                "debounce_seconds": 0.5,
                "update_interval_frames": 1,
                "frames_per_minute": 2,
                "analysis_window_minutes": 15,
                "idle_threshold_minutes": 15,
                "notify_on_idle": true,
                "alert_cooldown_minutes": 30,
                "emit_state_transitions": false,
                "emit_periodic_updates": false
            }
        }
        """
        looms_raw = config.get("looms", [])
        if not looms_raw:
            raise ValueError("At least one loom must be configured")

        self.looms: List[Dict[str, Any]] = []
        for idx, loom in enumerate(looms_raw):
            loom_id = loom.get("loom_id")
            if not loom_id:
                loom_id = f"loom-{idx+1:02d}"

            motion_roi = loom.get("motion_roi")
            if not motion_roi or len(motion_roi) != 4:
                raise ValueError(
                    f"Invalid motion_roi for loom '{loom_id}': "
                    f"must be [x1, y1, x2, y2] array with 4 integers"
                )

            try:
                motion_roi = [int(coord) for coord in motion_roi]
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid motion_roi for loom '{loom_id}': "
                    f"coordinates must be integers"
                )

            x1, y1, x2, y2 = motion_roi
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"Invalid motion_roi for loom '{loom_id}': "
                    f"x2 must be > x1 and y2 must be > y1"
                )

            self.looms.append({
                "loom_id": loom_id,
                "motion_roi": motion_roi,
                "name": loom.get("name", loom_id),
                "position": loom.get("position"),
                "line_number": loom.get("line_number"),
                "metadata": loom.get("metadata")
            })

        config_params = config.get("config", {})

        # MOG2 background subtractor (industrial-grade motion)
        self.mog2_history = int(config_params.get("mog2_history", 500))
        self.mog2_var_threshold = int(config_params.get("mog2_var_threshold", 16))
        self.mog2_detect_shadows = bool(config_params.get("mog2_detect_shadows", False))
        self.preprocess_blur_ksize = int(config_params.get("preprocess_blur_ksize", 5))  # odd, 0 = no blur
        self.morph_ksize = int(config_params.get("morph_ksize", 3))  # erosion/dilation kernel

        # Rolling window + hysteresis (state from avg motion over window)
        self.rolling_window_seconds = float(config_params.get("rolling_window_seconds", 15.0))
        self.motion_ratio_running = float(config_params.get("motion_ratio_running", 0.02))
        self.motion_ratio_stopped = float(config_params.get("motion_ratio_stopped", 0.005))
        self.debounce_seconds = float(config_params.get("debounce_seconds", 5.0))
        self.update_interval_frames = int(config_params.get("update_interval_frames", 0))  # 0 = from frames_per_minute

        # Idle alert
        self.frames_per_minute = int(config_params.get("frames_per_minute", 5))
        self.analysis_window_minutes = float(config_params.get("analysis_window_minutes", 1.0))
        self.idle_threshold_minutes = float(config_params.get("idle_threshold_minutes", 15.0))
        self.notify_on_idle = bool(config_params.get("notify_on_idle", True))
        self.alert_cooldown_minutes = float(config_params.get("alert_cooldown_minutes", 30.0))

        self.emit_state_transitions = bool(config_params.get("emit_state_transitions", False))
        self.emit_periodic_updates = bool(config_params.get("emit_periodic_updates", False))

        # Validation
        if self.mog2_history < 10:
            raise ValueError("mog2_history must be >= 10")
        if self.mog2_var_threshold < 1:
            raise ValueError("mog2_var_threshold must be >= 1")
        if self.rolling_window_seconds <= 0.0:
            raise ValueError("rolling_window_seconds must be > 0")
        if not 0.0 < self.motion_ratio_stopped < self.motion_ratio_running <= 1.0:
            raise ValueError("motion_ratio_stopped must be < motion_ratio_running and in (0, 1]")
        if self.debounce_seconds < 0.0:
            raise ValueError("debounce_seconds must be >= 0.0")
        if self.frames_per_minute < 1:
            raise ValueError("frames_per_minute must be >= 1")
        if self.analysis_window_minutes <= 0.0:
            raise ValueError("analysis_window_minutes must be > 0")
        if self.idle_threshold_minutes <= 0.0:
            raise ValueError("idle_threshold_minutes must be > 0")
        if self.alert_cooldown_minutes < 0.0:
            raise ValueError("alert_cooldown_minutes must be >= 0")
        if self.update_interval_frames < 0:
            raise ValueError("update_interval_frames must be >= 0")
