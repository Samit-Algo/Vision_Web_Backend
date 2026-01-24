"""
Weapon Detection Scenario
-------------------------

Main scenario class that orchestrates weapon detection using:
1. Pose extraction
2. Temporal buffering
3. Arm posture analysis
4. VLM confirmation
"""

from typing import List, Optional, Dict, Any
import os

from app.processing.scenarios.contracts import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent
)
from app.processing.scenarios.registry import register_scenario
from app.processing.scenarios.weapon_detection.config import WeaponDetectionConfig
from app.processing.scenarios.weapon_detection.state import WeaponDetectionState
from app.processing.scenarios.weapon_detection.pose_extractor import extract_pose_frame
from app.processing.scenarios.weapon_detection.posture_analyzer import analyze_arm_posture
from app.processing.scenarios.weapon_detection.vlm_handler import (
    should_call_vlm,
    call_vlm,
    get_person_key
)
from app.infrastructure.external.groq_vlm_service import GroqVLMService


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
    
    def _get_vlm_service(self) -> GroqVLMService:
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
        
        # Step 1: Extract pose data from detections
        pose_frame = extract_pose_frame(frame_context)
        if not pose_frame:
            return events  # No pose data, skip
        
        # Step 2: Buffer pose frames
        self.state.add_pose_frame(pose_frame)
        
        # Step 3: Analyze arm posture (if buffer is full)
        if len(self.state.pose_buffer) >= self.config_obj.temporal_consistency_frames:
            analyses = analyze_arm_posture(
                self.state.pose_buffer,
                self.config_obj.temporal_consistency_frames,
                self.config_obj.arm_angle_threshold
            )
            self.state.pending_analyses.extend(analyses)
        
        # Step 4: Call VLM for suspicious postures (non-blocking, throttled)
        vlm_service = self._get_vlm_service()
        for analysis in list(self.state.pending_analyses):
            if should_call_vlm(analysis, self.state, self.config_obj.vlm_throttle_seconds):
                print(f"[WeaponDetectionScenario] 🔍 Calling VLM for person {analysis.person_index} (frame {frame_context.frame_index}, arm_angle: {analysis.arm_angle:.1f}°)")
                
                vlm_result = call_vlm(
                    analysis,
                    frame_context.frame,
                    self.state,
                    vlm_service,
                    self.config_obj.vlm_confidence_threshold,
                    self.config_obj.vlm_frames_dir
                )
                
                if vlm_result and vlm_result.weapon_detected:
                    # Check if we've already emitted for this person recently
                    person_key = get_person_key(analysis.person_index, analysis.box)
                    last_emitted = self.state.emitted_events.get(person_key)
                    
                    # Emit event only if not recently emitted (avoid spam)
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
                                "vlm_response": vlm_result.vlm_response
                            },
                            detection_indices=[vlm_result.person_index],
                            timestamp=vlm_result.timestamp,
                            frame_index=vlm_result.frame_index
                        )
                        events.append(event)
                        self.state.emitted_events[person_key] = frame_context.timestamp
        
        # Cleanup: Remove old data
        self.state.cleanup_old_data(frame_context.timestamp)
        
        return events
    
    def reset(self) -> None:
        """Reset scenario state."""
        self.state.reset()
        self._vlm_service = None
