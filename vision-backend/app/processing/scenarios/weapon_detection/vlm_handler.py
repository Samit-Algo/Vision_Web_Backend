"""
VLM Handler
-----------

Handles VLM calls for weapon confirmation.
Manages caching, throttling, and frame saving.
"""

from typing import Optional, Dict, Any
import os
import cv2
import numpy as np

from app.processing.scenarios.weapon_detection.types import (
    ArmPostureAnalysis,
    VLMConfirmation
)
from app.processing.scenarios.weapon_detection.state import WeaponDetectionState
from app.infrastructure.external.groq_vlm_service import GroqVLMService


WEAPON_DETECTION_PROMPT = """Analyze these images (multiple frames from a video sequence) and determine if there is a weapon visible (gun, knife, sword, or any other weapon) in ANY of the frames.

The images are sequential frames showing the same person. Look across all frames to detect weapons - a weapon may be visible in some frames but not others.

Respond ONLY with a valid JSON object in this exact format:
{
    "weapon_detected": true or false,
    "weapon_type": "gun" or "knife" or "sword" or "other" or null,
    "confidence": 0.0 to 1.0,
    "description": "Brief description of what you see across the frames"
}

Be very strict - only return true if you are confident a weapon is actually visible in at least one frame. False positives are worse than false negatives."""


def get_person_key(person_index: int, box: list[float]) -> str:
    """
    Generate a unique key for a person (for caching/throttling).
    
    Args:
        person_index: Person index
        box: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Unique string key
    """
    box_str = ",".join(f"{v:.1f}" for v in box[:4])
    return f"person_{person_index}_{box_str}"


def should_call_vlm(
    analysis: ArmPostureAnalysis,
    state: WeaponDetectionState,
    vlm_throttle_seconds: float
) -> bool:
    """
    Check if VLM should be called for this analysis.
    
    Throttles calls to avoid blocking pipeline.
    
    Args:
        analysis: Arm posture analysis
        state: Weapon detection state
        vlm_throttle_seconds: Minimum seconds between VLM calls for same person
    
    Returns:
        True if VLM should be called
    """
    person_key = get_person_key(analysis.person_index, analysis.box)
    
    # Check cache first
    if person_key in state.vlm_cache:
        cached = state.vlm_cache[person_key]
        # Use cached result if recent (within 5 seconds)
        if (analysis.timestamp - cached.timestamp).total_seconds() < 5.0:
            return False
    
    # Check throttle
    last_call = state.last_vlm_call_time.get(person_key, 0.0)
    current_time = analysis.timestamp.timestamp()
    if current_time - last_call < vlm_throttle_seconds:
        return False
    
    return True


def save_vlm_frame(
    frame: np.ndarray,
    analysis: ArmPostureAnalysis,
    frames_dir: str
) -> str:
    """
    Save frame before sending to VLM.
    
    Args:
        frame: Full frame image
        analysis: Arm posture analysis with bounding box
        frames_dir: Directory to save frames
    
    Returns:
        Path to saved frame file
    """
    frame_filename = f"vlm_frame_{analysis.frame_index}_person_{analysis.person_index}_{int(analysis.timestamp.timestamp())}.jpg"
    frame_path = os.path.join(frames_dir, frame_filename)
    
    # Crop person region from frame
    x1, y1, x2, y2 = [int(coord) for coord in analysis.box[:4]]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    if x2 > x1 and y2 > y1:
        person_crop = frame[y1:y2, x1:x2]
        cv2.imwrite(frame_path, person_crop)
        print(f"[WeaponDetectionScenario] 💾 Saved VLM frame: {frame_path} (person {analysis.person_index}, box: [{x1},{y1},{x2},{y2}])")
    else:
        # If box is invalid, save full frame
        cv2.imwrite(frame_path, frame)
        print(f"[WeaponDetectionScenario] 💾 Saved VLM frame (full): {frame_path} (person {analysis.person_index}, invalid box)")
    
    return frame_path


def call_vlm(
    analysis: ArmPostureAnalysis,
    frames: list[np.ndarray],
    state: WeaponDetectionState,
    vlm_service: GroqVLMService,
    vlm_confidence_threshold: float,
    frames_dir: str
) -> Optional[VLMConfirmation]:
    """
    Call VLM model to confirm weapon presence using multiple buffered frames.
    
    Non-blocking: Returns cached result if available, otherwise calls VLM.
    
    Args:
        analysis: Arm posture analysis
        frames: List of frame images (buffered frames that showed suspicious posture)
        state: Weapon detection state
        vlm_service: VLM service instance
        vlm_confidence_threshold: Minimum confidence threshold
        frames_dir: Directory to save frames
    
    Returns:
        VLMConfirmation if weapon detected, None otherwise
    """
    person_key = get_person_key(analysis.person_index, analysis.box)
    
    # Check cache first
    if person_key in state.vlm_cache:
        cached = state.vlm_cache[person_key]
        if (analysis.timestamp - cached.timestamp).total_seconds() < 5.0:
            return cached
    
    # Update throttle
    state.last_vlm_call_time[person_key] = analysis.timestamp.timestamp()
    
    try:
        # Save all frames before sending to VLM
        for idx, frame in enumerate(frames):
            # Create a temporary analysis with frame index for saving
            temp_analysis = ArmPostureAnalysis(
                person_index=analysis.person_index,
                box=analysis.box,
                arm_raised=analysis.arm_raised,
                arm_angle=analysis.arm_angle,
                confidence=analysis.confidence,
                timestamp=analysis.timestamp,
                frame_index=analysis.frame_index + idx - len(frames) + 1
            )
            save_vlm_frame(frame, temp_analysis, frames_dir)
        
        # Call VLM with multiple frames
        vlm_result = vlm_service.analyze_images(
            frames,
            prompt=WEAPON_DETECTION_PROMPT,
            crop_box=analysis.box,
            temperature=0.1,
            max_tokens=500
        )
        
        # Parse weapon detection response
        parsed_json = vlm_result.get("parsed_json")
        if parsed_json:
            weapon_detected = bool(parsed_json.get("weapon_detected", False))
            weapon_type = parsed_json.get("weapon_type")
            confidence = float(parsed_json.get("confidence", 0.0))
            description = str(parsed_json.get("description", ""))
        else:
            # Fallback: try to infer from text content
            content_lower = vlm_result.get("content", "").lower()
            weapon_detected = any(keyword in content_lower for keyword in [
                "weapon", "gun", "knife", "sword", "firearm", "pistol", "rifle"
            ])
            weapon_type = "unknown" if weapon_detected else None
            confidence = 0.5 if weapon_detected else 0.0
            description = vlm_result.get("content", "")
        
        # Clamp confidence to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))
        
        # Check if weapon detected and confidence meets threshold
        weapon_confirmed = (
            weapon_detected and
            confidence >= vlm_confidence_threshold
        )
        
        # Create VLMConfirmation
        vlm_confirmation = VLMConfirmation(
            person_index=analysis.person_index,
            box=analysis.box,
            weapon_detected=weapon_confirmed,
            weapon_type=weapon_type if weapon_confirmed else None,
            confidence=confidence,
            description=description if weapon_confirmed else None,
            vlm_response=vlm_result.get("raw_response", {}),
            timestamp=analysis.timestamp,
            frame_index=analysis.frame_index
        )
        print(f"[WeaponDetectionScenario] VLM result: {vlm_confirmation}")
        
        # Cache the result
        state.vlm_cache[person_key] = vlm_confirmation
        
        # Return confirmation only if weapon detected
        return vlm_confirmation if weapon_detected else None
        
    except Exception as e:
        # Log error but don't crash the pipeline
        print(f"[WeaponDetectionScenario] ⚠️ Error calling VLM: {e}")
        return None
