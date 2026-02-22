"""
Weapon Detection Scenario
--------------------------
Pose keypoints → temporal buffer → arm posture analysis → VLM confirmation → weapon_detected event.
Only emits when VLM confirms. Pipeline draws person + keypoints for overlay.

Code layout:
  - WeaponDetectionScenario: __init__ (config, state buffer, VLM dir), get_vlm_service, process (extract pose → buffer → deferred VLM → events), get_overlay_data.
"""

# -------- Imports --------
import os
from typing import Any, Dict, List, Optional

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
    OverlayData
)
from app.processing.vision_tasks.task_lookup import register_scenario
from .config import WeaponDetectionConfig
from .state import WeaponDetectionState
from .pose_extractor import extract_pose_frame
from .posture_analyzer import analyze_arm_posture, check_arm_raised
from .vlm_handler import (
    should_call_vlm,
    call_vlm,
    get_person_key
)
from app.infrastructure.external.groq_vlm_service import GroqVLMService


# ========== Scenario: Weapon detection (pose → buffer → VLM confirm) ==========

@register_scenario("weapon_detection")
class WeaponDetectionScenario(BaseScenario):
    """
    Weapon detection scenario using pose → temporal → VLM pipeline.
    
    Flow:
    1. Extract pose keypoints from detections
    2. Buffer 4-5 frames
    3. Analyze arm posture (temporal consistency)
    4. If suspicious posture detected, call VLM
    5. Emit weapon_detected event when VLM confirms
    """
    
    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        
        # Load configuration
        self.config_obj = WeaponDetectionConfig(config)
        
        # Initialize state
        self.state = WeaponDetectionState(self.config_obj.buffer_size)
        
        # VLM service (lazy-loaded)
        self._vlm_service: Optional[GroqVLMService] = None
        
        # Create frames directory
        os.makedirs(self.config_obj.vlm_frames_dir, exist_ok=True)
    
    def get_vlm_service(self) -> GroqVLMService:
        """Get or create VLM service instance (lazy initialization)."""
        if self._vlm_service is None:
            self._vlm_service = GroqVLMService()
        return self._vlm_service
    
    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process a single frame.
        
        Only processes if pose keypoints are available.
        Returns empty list immediately if no pose data.
        
        Returns events only when VLM confirms weapon.
        """
        events = []
        # --- Extract pose; skip if no keypoints ---
        pose_frame = extract_pose_frame(frame_context)
        if not pose_frame:
            return events
        self.state.add_pose_frame(pose_frame)
        # --- Process deferred VLM: when we have 3 frames (before, suspicious, after), call VLM ---
        vlm_service = self.get_vlm_service()
        to_remove_deferred = []
        for analysis, suspicious_frame_index in list(self.state.deferred_vlm):
            if frame_context.frame_index != suspicious_frame_index + 1:
                continue
            # We have N+1 = current frame; buffer has ... N-1, N, N+1 (current was just added in Step 2)
            if len(self.state.pose_buffer) < 3:
                continue
            before_frame = self.state.pose_buffer[-3].frame
            suspicious_frame = self.state.pose_buffer[-2].frame
            after_frame = self.state.pose_buffer[-1].frame
            three_frames = [before_frame, suspicious_frame, after_frame]
            print(f"[WeaponDetectionScenario] 🔍 Calling VLM for person {analysis.person_index} with 3 frames (before={suspicious_frame_index - 1}, suspicious={suspicious_frame_index}, after={suspicious_frame_index + 1})")
            vlm_result = call_vlm(
                analysis,
                three_frames,
                self.state,
                vlm_service,
                self.config_obj.vlm_confidence_threshold,
                self.config_obj.vlm_frames_dir,
                weapon_types=self.config_obj.weapon_types,
            )
            to_remove_deferred.append((analysis, suspicious_frame_index))
            if vlm_result and vlm_result.weapon_detected:
                person_key = get_person_key(analysis.person_index, analysis.box)
                last_emitted = self.state.emitted_events.get(person_key)
                if not last_emitted or (frame_context.timestamp - last_emitted).total_seconds() > 5.0:
                    event = ScenarioEvent(
                        event_type="weapon_detected",
                        label=f"Weapon detected: {vlm_result.weapon_type} (confidence: {vlm_result.confidence:.2f})",
                        confidence=vlm_result.confidence,
                        metadata={
                            "weapon_type": vlm_result.weapon_type,
                            "person_index": vlm_result.person_index,
                            "box": vlm_result.box,
                            "arm_angle": analysis.arm_angle,
                            "vlm_description": vlm_result.description,
                            "vlm_response": vlm_result.vlm_response
                        },
                        detection_indices=[vlm_result.person_index],
                        timestamp=vlm_result.timestamp,
                        frame_index=vlm_result.frame_index
                    )
                    events.append(event)
                    self.state.emitted_events[person_key] = frame_context.timestamp
        for item in to_remove_deferred:
            self.state.deferred_vlm.remove(item)
        
        # Step 4: Analyze arm posture (if buffer is full) and defer VLM for next frame
        if len(self.state.pose_buffer) >= self.config_obj.temporal_consistency_frames:
            analyses = analyze_arm_posture(
                self.state.pose_buffer,
                self.config_obj.temporal_consistency_frames,
                self.config_obj.arm_angle_threshold,
                self.config_obj.kp_confidence_threshold,
                require_shoulder_height=self.config_obj.require_shoulder_height,
            )
            self.state.pending_analyses.extend(analyses)
        
        # Step 5: For new suspicious postures, defer VLM (call on next frame with [N-1, N, N+1])
        deferred_person_keys = {get_person_key(a.person_index, a.box) for a, _ in self.state.deferred_vlm}
        for analysis in list(self.state.pending_analyses):
            if not analysis.arm_raised:
                continue
            person_key = get_person_key(analysis.person_index, analysis.box)
            if person_key in deferred_person_keys:
                continue
            if not should_call_vlm(analysis, self.state, self.config_obj.vlm_throttle_seconds):
                continue
            self.state.deferred_vlm.append((analysis, analysis.frame_index))
            deferred_person_keys.add(person_key)
            print(f"[WeaponDetectionScenario] ⏳ Deferred VLM for person {analysis.person_index} (suspicious at frame {analysis.frame_index}); will send 3 frames on next frame.")
        
        # Cleanup: Remove old data
        self.state.cleanup_old_data(frame_context.timestamp)
        
        return events
    
    def reset(self) -> None:
        """Reset scenario state."""
        self.state.reset()
        self._vlm_service = None

    def requires_yolo_detections(self) -> bool:
        """This scenario requires YOLO pose detections."""
        return True

    def get_overlay_data(self, frame_context: Optional[ScenarioFrameContext] = None) -> List[OverlayData]:
        """Get overlay data for visualization. frame_context is optional (overlay uses scenario state only)."""
        overlays = []
        # Visualize suspicious arm postures
        for analysis in self.state.pending_analyses:
            if analysis.arm_raised:
                label = f"Suspicious Pose ({analysis.arm_angle:.1f}deg)"
                overlays.append(OverlayData(box=analysis.box, label=label, color=(0, 255, 255)))  # Cyan for suspicious
        
        # Visualize confirmed weapon detections
        for person_key, timestamp in self.state.emitted_events.items():
            # Find the analysis that triggered this
            # This is a bit inefficient but works for visualization
            for analysis in self.state.pending_analyses:
                if get_person_key(analysis.person_index, analysis.box) == person_key:
                    label = "WEAPON DETECTED"
                    overlays.append(OverlayData(box=analysis.box, label=label, color=(0, 0, 255))) # Red for confirmed
                    break

        return overlays
