"""
Loom Machine State Configuration
--------------------------------

Handles configuration parsing and validation for loom machine state scenario.
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
                "motion_threshold": 0.15,
                "optical_flow_threshold": 2.0,
                "temporal_consistency_seconds": 2.0,
                "debounce_seconds": 0.5,
                "update_interval_frames": 1
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

        self.motion_threshold = float(config_params.get("motion_threshold", 0.02))
        self.optical_flow_threshold = float(config_params.get("optical_flow_threshold", 0.5))
        self.temporal_consistency_seconds = float(config_params.get("temporal_consistency_seconds", 2.0))
        self.temporal_consistency_ratio = float(config_params.get("temporal_consistency_ratio", 0.4))
        self.debounce_seconds = float(config_params.get("debounce_seconds", 0.5))
        self.update_interval_frames = int(config_params.get("update_interval_frames", 1))

        self.emit_state_transitions = bool(config_params.get("emit_state_transitions", False))
        self.emit_periodic_updates = bool(config_params.get("emit_periodic_updates", False))

        if not 0.0 <= self.motion_threshold <= 1.0:
            raise ValueError("motion_threshold must be between 0.0 and 1.0")

        if self.optical_flow_threshold < 0.0:
            raise ValueError("optical_flow_threshold must be >= 0.0")

        if not 0.0 <= self.temporal_consistency_ratio <= 1.0:
            raise ValueError("temporal_consistency_ratio must be between 0.0 and 1.0")

        if self.temporal_consistency_seconds < 0.0:
            raise ValueError("temporal_consistency_seconds must be >= 0.0")

        if self.debounce_seconds < 0.0:
            raise ValueError("debounce_seconds must be >= 0.0")

        if self.update_interval_frames < 1:
            raise ValueError("update_interval_frames must be >= 1")
