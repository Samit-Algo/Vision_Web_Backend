"""
Loom Machine State Scenario
---------------------------

Main scenario class that determines RUNNING/STOPPED state per loom
using motion-based analysis on per-loom ROIs.
"""

from typing import List, Optional, Dict, Any
import numpy as np

from app.processing.scenarios.contracts import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent
)
from app.processing.scenarios.registry import register_scenario
from app.processing.scenarios.loom_machine_state.config import LoomMachineStateConfig
from app.processing.scenarios.loom_machine_state.state import LoomMachineStateManager
from app.processing.scenarios.loom_machine_state.motion_detector import (
    detect_motion,
    extract_roi
)


@register_scenario("loom_machine_state")
class LoomMachineStateScenario(BaseScenario):
    """
    Loom machine state detection scenario.
    
    Determines RUNNING/STOPPED state per loom using motion-based analysis.
    One camera may contain multiple looms, each with its own motion ROI and independent state.
    
    Flow:
    1. Extract ROI for each loom from frame
    2. Compute motion (frame diff + optical flow)
    3. Update motion history buffer
    4. Check temporal consistency
    5. Update state machine (if needed)
    6. Store state (no events by default)
    """
    
    def requires_yolo_detections(self) -> bool:
        """Loom scenario doesn't need YOLO - only uses raw frames for motion detection."""
        return False
    
    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        
        # Load configuration
        self.config_obj = LoomMachineStateConfig(config)
        
        # Get FPS from pipeline context
        fps = pipeline_context.fps
        
        # Initialize state manager
        self.state_manager = LoomMachineStateManager(self.config_obj, fps)
        
        # Store previous ROI frames for each loom (for frame difference)
        self.previous_rois: Dict[str, Optional[np.ndarray]] = {
            loom["loom_id"]: None
            for loom in self.config_obj.looms
        }
        
        # Frame counter for update interval
        self.frame_counter = 0
        
        # Track previous state for each loom (to detect transitions)
        self.previous_states: Dict[str, str] = {
            loom["loom_id"]: "UNKNOWN"
            for loom in self.config_obj.looms
        }
        
        print(
            f"[LoomMachineStateScenario] Initialized with {len(self.config_obj.looms)} loom(s): "
            f"{[loom['loom_id'] for loom in self.config_obj.looms]}"
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
        
        # Check if we should process this frame (update interval)
        self.frame_counter += 1
        if self.frame_counter % self.config_obj.update_interval_frames != 0:
            return events  # Skip this frame
        
        # Process each loom independently
        for loom in self.config_obj.looms:
            loom_id = loom["loom_id"]
            motion_roi = loom["motion_roi"]
            
            # Get previous ROI frame
            previous_roi = self.previous_rois[loom_id]
            
            # Detect motion
            motion_analysis = detect_motion(
                frame=frame,
                roi=motion_roi,
                previous_roi=previous_roi,
                motion_threshold=self.config_obj.motion_threshold,
                optical_flow_threshold=self.config_obj.optical_flow_threshold,
                loom_id=loom_id,
                frame_index=frame_index,
                timestamp=timestamp
            )
            
            # Update state manager and check for transitions
            previous_state = self.state_manager.add_motion_analysis(motion_analysis)
            
            # Debug logging (every 10 frames)
            if frame_index % 10 == 0:
                print(
                    f"[LoomMachineStateScenario] {loom_id}: "
                    f"motion_energy={motion_analysis.motion_energy:.4f} "
                    f"(threshold={self.config_obj.motion_threshold:.4f}), "
                    f"optical_flow={motion_analysis.optical_flow_magnitude}, "
                    f"detected={motion_analysis.motion_detected}, "
                    f"state={self.state_manager.get_loom_state(loom_id).current_state if self.state_manager.get_loom_state(loom_id) else 'N/A'}"
                )
            
            # Store current ROI for next frame
            current_roi = extract_roi(frame, motion_roi)
            self.previous_rois[loom_id] = current_roi
            
            # Get current loom state
            loom_state = self.state_manager.get_loom_state(loom_id)
            current_state = loom_state.current_state
            
            # Emit state transition event if configured and state changed
            if self.config_obj.emit_state_transitions and previous_state is not None:
                # State transition occurred (previous_state returned from add_motion_analysis)
                # Only emit if transitioning from a known state (not UNKNOWN on first detection)
                if previous_state != "UNKNOWN" and previous_state != current_state:
                    # Get loom name for better event label
                    loom_name = next(
                        (loom.get("name", loom_id) for loom in self.config_obj.looms if loom["loom_id"] == loom_id),
                        loom_id
                    )
                    
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
                            "motion_energy": loom_state.last_motion_energy,
                            "optical_flow_magnitude": loom_state.last_optical_flow_magnitude,
                            "transition_timestamp": timestamp.isoformat()
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
                            "motion_energy": loom_state.last_motion_energy,
                            "optical_flow_magnitude": loom_state.last_optical_flow_magnitude
                        },
                        detection_indices=[],
                        timestamp=timestamp,
                        frame_index=frame_index
                    )
                    events.append(event)
        
        # Cleanup old data
        self.state_manager.cleanup_old_data(timestamp)
        
        # Store state in scenario's internal state (for other scenarios to access)
        self._update_internal_state()
        
        return events
    
    def _update_internal_state(self) -> None:
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
                "motion_threshold": self.config_obj.motion_threshold,
                "optical_flow_threshold": self.config_obj.optical_flow_threshold,
                "temporal_consistency_seconds": self.config_obj.temporal_consistency_seconds
            }
        }
    
    def get_overlay_data(self) -> Optional[Dict[str, Any]]:
        """Get ROI boxes with state labels for overlay visualization."""
        all_states = self.state_manager.get_all_states()
        boxes = []
        labels = []
        colors = []
        
        for loom in self.config_obj.looms:
            loom_id = loom["loom_id"]
            motion_roi = loom["motion_roi"]
            loom_state = all_states.get(loom_id)
            
            if loom_state:
                # Add ROI box
                boxes.append(motion_roi)  # [x1, y1, x2, y2]
                
                # Create label with state
                state = loom_state.current_state
                loom_name = loom.get("name", loom_id)
                label = f"{loom_name}: {state}"
                labels.append(label)
                
                # Color: green for RUNNING, red for STOPPED, yellow for UNKNOWN
                if state == "RUNNING":
                    colors.append("#00ff00")  # Green
                elif state == "STOPPED":
                    colors.append("#ff0000")  # Red
                else:
                    colors.append("#ffff00")  # Yellow
        
        if not boxes:
            return None
        
        return {
            "boxes": boxes,
            "labels": labels,
            "colors": colors
        }
    
    def reset(self) -> None:
        """Reset scenario state."""
        self.state_manager.reset()
        self.previous_rois = {
            loom["loom_id"]: None
            for loom in self.config_obj.looms
        }
        self.previous_states = {
            loom["loom_id"]: "UNKNOWN"
            for loom in self.config_obj.looms
        }
        self.frame_counter = 0
        self._state.clear()
