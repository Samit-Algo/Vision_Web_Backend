"""
Loom Machine State Scenario
----------------------------
No YOLO; raw frames only. MOG2 background subtraction + motion ratio over rolling window.
RUNNING when motion > threshold, STOPPED when below. Idle alert after idle_threshold_minutes.
Overlay: polygons + colors (green=RUNNING, red=STOPPED) per loom ROI.

Code layout:
  - LoomMachineStateScenario: requires_yolo_detections=False, __init__ (config, state_manager, MOG2 per loom), process, get_overlay_data, reset.
"""

# -------- Imports --------
from typing import Any, Dict, List, Optional

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from .config import LoomMachineStateConfig
from .state import LoomMachineStateManager
from .motion_detector import create_mog2_subtractor, detect_motion


# ========== Scenario: Loom machine state (MOG2 motion → RUNNING/STOPPED) ==========

@register_scenario("loom_machine_state")
class LoomMachineStateScenario(BaseScenario):
    """
    Industrial loom state: MOG2 + 15s rolling average + hysteresis.
    Green=RUNNING, Red=STOPPED. Idle alert after idle_threshold_minutes.
    """

    def requires_yolo_detections(self) -> bool:
        return False

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = LoomMachineStateConfig(config)
        fps = getattr(pipeline_context, "fps", None) or 5
        if self.config_obj.update_interval_frames == 0:
            self.config_obj.update_interval_frames = max(
                1, (fps * 60) // self.config_obj.frames_per_minute
            )
        self.state_manager = LoomMachineStateManager(self.config_obj, fps)
        self.mog2_subtractors: Dict[str, Any] = {
            loom["loom_id"]: create_mog2_subtractor(
                history=self.config_obj.mog2_history,
                var_threshold=self.config_obj.mog2_var_threshold,
                detect_shadows=self.config_obj.mog2_detect_shadows,
            )
            for loom in self.config_obj.looms
        }
        self.frame_counter = 0
        self.previous_states: Dict[str, str] = {
            loom["loom_id"]: "UNKNOWN" for loom in self.config_obj.looms
        }
        print(
            f"[LoomMachineState] looms={[l['loom_id'] for l in self.config_obj.looms]}, "
            f"fpm={self.config_obj.frames_per_minute}, roll_win={self.config_obj.rolling_window_seconds}s, "
            f"idle_threshold_min={self.config_obj.idle_threshold_minutes}"
        )
    
    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process a single frame.
        
        Returns empty list by default (state-only scenario).
        Can emit events if configured (state transitions, periodic updates).
        
        Args:
            frame_context: Per-frame context from pipeline
        
        Returns:
            List of scenario events (typically empty)
        """
        events = []
        frame = frame_context.frame
        frame_index = frame_context.frame_index
        timestamp = frame_context.timestamp
        # --- Process at update_interval_frames (e.g. once per N frames) ---
        self.frame_counter += 1
        if self.frame_counter % self.config_obj.update_interval_frames != 0:
            return events  # Skip this frame
        
        for loom in self.config_obj.looms:
            loom_id = loom["loom_id"]
            motion_roi = loom["motion_roi"]
            mog2 = self.mog2_subtractors[loom_id]
            motion_analysis = detect_motion(
                frame=frame,
                roi=motion_roi,
                mog2=mog2,
                motion_ratio_stopped=self.config_obj.motion_ratio_stopped,
                preprocess_blur_ksize=self.config_obj.preprocess_blur_ksize,
                morph_ksize=self.config_obj.morph_ksize,
                loom_id=loom_id,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            previous_state = self.state_manager.add_motion_analysis(motion_analysis)
            loom_state = self.state_manager.get_loom_state(loom_id)
            current_state = loom_state.current_state if loom_state else "UNKNOWN"

            # Emit state transition event if configured and state changed
            loom_name = loom.get("name", loom_id)
            if self.config_obj.emit_state_transitions and previous_state is not None:
                # State transition occurred (previous_state returned from add_motion_analysis)
                # Only emit if transitioning from a known state (not UNKNOWN on first detection)
                if previous_state != "UNKNOWN" and previous_state != current_state:
                    event = ScenarioEvent(
                        event_type="loom_state_transition",
                        label=f"Loom '{loom_name}' ({loom_id}) changed from {previous_state} to {current_state}",
                        confidence=loom_state.confidence,
                        metadata={
                            "loom_id": loom_id,
                            "loom_name": loom_name,
                            "previous_state": previous_state,
                            "current_state": current_state,
                            "state_duration_seconds": loom_state.state_duration_seconds,
                            "motion_ratio": loom_state.last_motion_energy,
                            "transition_timestamp": timestamp.isoformat(),
                        },
                        detection_indices=[],
                        timestamp=timestamp,
                        frame_index=frame_index
                    )
                    events.append(event)
                    
                    print(
                        f"[LoomMachineStateScenario] 📢 Emitted state transition event: "
                        f"Loom '{loom_id}' {previous_state} → {current_state}"
                    )
            
            # Idle alert: notify when machine idle for >= idle_threshold_minutes
            idle_alert = self.state_manager.check_idle_alert(loom_id, loom_name, timestamp)
            if idle_alert:
                events.append(
                    ScenarioEvent(
                        event_type="loom_idle_alert",
                        label=f"Machine '{loom_name}' ({loom_id}) has been idle for {idle_alert['idle_duration_minutes']:.1f} minutes",
                        confidence=1.0,
                        metadata={
                            "loom_id": loom_id,
                            "loom_name": loom_name,
                            "idle_duration_minutes": idle_alert["idle_duration_minutes"],
                            "idle_since": idle_alert.get("idle_since"),
                            "report": {
                                "machine": loom_name,
                                "idle_minutes": round(idle_alert["idle_duration_minutes"], 1),
                            },
                        },
                        detection_indices=[],
                        timestamp=timestamp,
                        frame_index=frame_index
                    )
                )
                print(
                    f"[LoomMachineStateScenario] 🚨 Idle alert: {loom_name} ({loom_id}) idle "
                    f"for {idle_alert['idle_duration_minutes']:.1f} min"
                )
            
            # Update previous state tracking
            self.previous_states[loom_id] = current_state
            
            # Emit periodic update if configured
            if self.config_obj.emit_periodic_updates:
                # Emit every N seconds (e.g., every 10 seconds)
                if frame_index % (self.pipeline_context.fps * 10) == 0:
                    event = ScenarioEvent(
                        event_type="loom_state_update",
                        label=f"Loom {loom_id} ({loom_state.current_state})",
                        confidence=loom_state.confidence,
                        metadata={
                            "loom_id": loom_id,
                            "state": loom_state.current_state,
                            "state_duration_seconds": loom_state.state_duration_seconds,
                            "motion_ratio": loom_state.last_motion_energy,
                        },
                        detection_indices=[],
                        timestamp=timestamp,
                        frame_index=frame_index
                    )
                    events.append(event)
        
        # Cleanup old data
        self.state_manager.cleanup_old_data(timestamp)
        
        # Store state in scenario's internal state (for other scenarios to access)
        self.update_internal_state()
        
        return events
    
    def update_internal_state(self) -> None:
        """Update scenario's internal state dictionary."""
        all_states = self.state_manager.get_all_states()
        
        self._state = {
            "looms": {
                loom_id: {
                    "current_state": state.current_state,
                    "state_since": state.state_since.isoformat() if state.state_since else None,
                    "state_duration_seconds": state.state_duration_seconds,
                    "confidence": state.confidence,
                    "last_motion_energy": state.last_motion_energy,
                    "last_optical_flow_magnitude": state.last_optical_flow_magnitude,
                    "last_updated": state.last_updated.isoformat()
                }
                for loom_id, state in all_states.items()
            },
            "config": {
                "rolling_window_seconds": self.config_obj.rolling_window_seconds,
                "motion_ratio_running": self.config_obj.motion_ratio_running,
                "motion_ratio_stopped": self.config_obj.motion_ratio_stopped,
            }
        }
    
    def get_overlay_data(self, frame_context=None) -> Optional[Dict[str, Any]]:
        """Get ROI polygons with colors for overlay visualization (no text labels). frame_context is optional for API compatibility."""
        from app.utils.datetime_utils import utc_now
        all_states = self.state_manager.get_all_states()
        polygons = []
        colors = []
        now = utc_now()

        for loom in self.config_obj.looms:
            loom_id = loom["loom_id"]
            motion_roi = loom["motion_roi"]  # [x1, y1, x2, y2]
            
            # Always show polygon, even if state is not initialized yet
            # Convert bounding box [x1, y1, x2, y2] to polygon [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x1, y1, x2, y2 = motion_roi
            polygon = [
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x2, y2],  # Bottom-right
                [x1, y2],  # Bottom-left
            ]
            polygons.append(polygon)

            # Get state if available, otherwise default to UNKNOWN
            loom_state = all_states.get(loom_id)
            if loom_state:
                state = loom_state.current_state
            else:
                state = "UNKNOWN"

            # Color based on state: Green for RUNNING, Red for STOPPED
            if state == "RUNNING":
                colors.append("#00ff00")  # Green
            elif state == "STOPPED":
                colors.append("#ff0000")  # Red
            else:
                colors.append("#808080")  # Gray for UNKNOWN

        if not polygons:
            return None

        return {
            "polygons": polygons,
            "colors": colors
        }
    
    def reset(self) -> None:
        """Reset scenario state. MOG2 subtractors are re-used (they retain background model)."""
        self.state_manager.reset()
        self.previous_states = {loom["loom_id"]: "UNKNOWN" for loom in self.config_obj.looms}
        self.frame_counter = 0
        self._state.clear()
