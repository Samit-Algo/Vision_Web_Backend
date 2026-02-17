"""
VLM Handler
-----------

Step 3: Send 3 frames to the VLM and ask: "Is this person climbing/jumping the wall?"

We only call the VLM when detection says "person above zone" (to save cost).
We send: frame before, frame when we flagged, frame after.
We parse the VLM reply: climbing_detected (true/false), confidence, description.
"""

import os
from typing import Optional, List
import cv2
import numpy as np

from app.processing.vision_tasks.tasks.wall_climb_detection.types import (
    WallClimbAnalysis,
    WallClimbVLMConfirmation,
)
from app.processing.vision_tasks.tasks.wall_climb_detection.state import WallClimbDetectionState
from app.infrastructure.external.groq_vlm_service import GroqVLMService


def build_wall_climb_detection_prompt() -> str:
    """
    Prompt for the VLM: we send 3 images (before, suspicious moment, after).
    Ask for JSON with climbing_detected, confidence, description.
    
    Detects all climbing behaviors: active climbing, preparation/ready-to-climb, jumping over, and suspicious positioning.
    """
    return """These 3 images are sequential video frames from a security camera: (1) before, (2) suspicious moment, (3) after.

Determine if the person is CLIMBING, PREPARING TO CLIMB, or JUMPING OVER a wall/fence/barrier.

**DETECT (climbing_detected = true):**
- Active climbing: upper body above barrier, arms gripping/pulling, legs pushing up, scaling posture
- Preparation: hands reaching up, crouching/jumping position, arms raised to grab, legs bent for thrust, assessing barrier
- Jumping: airborne over barrier, legs clearing height
- Suspicious: testing barrier stability/height, unusual positioning suggesting climbing intent

**IGNORE (climbing_detected = false):**
- Standing/walking past barrier (normal posture)
- Leaning against wall casually
- Looking/inspecting without climbing intent
- Body clearly below barrier with no upward movement

**ANALYSIS:**
Analyze body position, posture, and movement across all 3 frames. Look for progression: preparation → attempt → completion. Consider arm/leg positions relative to barrier and overall intent.

Respond ONLY with valid JSON:
{
    "climbing_detected": true or false,
    "confidence": 0.0 to 1.0,
    "description": "Brief description: position relative to barrier, posture, arm/leg positions, movement, reasoning"
}

**CONFIDENCE:** 0.9-1.0 (clear climbing), 0.7-0.89 (strong evidence/preparation), 0.5-0.69 (moderate), 0.0-0.49 (none/normal)

Be strict: only true if clear evidence of climbing/preparation/intent. False positives worse than false negatives."""


def should_call_vlm(
    analysis: WallClimbAnalysis,
    state: WallClimbDetectionState,
    vlm_throttle_seconds: float,
) -> bool:
    """
    Decide if we should call the VLM for this person now.
    - Don't call if we have a recent cached result (within 5 seconds).
    - Don't call if we called VLM for this person too recently (throttle).
    """
    track_id = analysis.track_id

    if track_id in state.vlm_cache:
        cached = state.vlm_cache[track_id]
        if (analysis.timestamp - cached.timestamp).total_seconds() < 5.0:
            print(f"[WallClimb VLM] ⏭️ should_call_vlm: skip (cached recent) track_id={track_id}")
            return False

    last_call = state.last_vlm_call_time.get(track_id, 0.0)
    current_time = analysis.timestamp.timestamp()
    if current_time - last_call < vlm_throttle_seconds:
        print(f"[WallClimb VLM] ⏭️ should_call_vlm: skip (throttle) track_id={track_id} last_call_ago={current_time - last_call:.1f}s")
        return False

    print(f"[WallClimb VLM] ✅ should_call_vlm: yes track_id={track_id}")
    return True


def save_vlm_frame(
    frame: np.ndarray,
    analysis: WallClimbAnalysis,
    frames_dir: str,
) -> str:
    """Save one frame to disk (optional, for debugging). Returns path."""
    os.makedirs(frames_dir, exist_ok=True)
    frame_filename = f"wall_climb_frame_{analysis.frame_index}_track_{analysis.track_id}_{int(analysis.timestamp.timestamp())}.jpg"
    frame_path = os.path.join(frames_dir, frame_filename)

    x1, y1, x2, y2 = [int(c) for c in analysis.box[:4]]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 > x1 and y2 > y1:
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(frame_path, crop)
    else:
        cv2.imwrite(frame_path, frame)

    return frame_path


def call_vlm(
    analysis: WallClimbAnalysis,
    frames: List[np.ndarray],
    state: WallClimbDetectionState,
    vlm_service: GroqVLMService,
    vlm_confidence_threshold: float,
    frames_dir: str,
) -> Optional[WallClimbVLMConfirmation]:
    """
    Send the 3 frames to the VLM and parse the response.

    - frames: [before_frame, suspicious_frame, after_frame]
    - We crop to the person's box so the VLM sees the person clearly.
    - We parse JSON: climbing_detected, confidence, description.
    - We only "confirm" climbing if climbing_detected is true and confidence >= threshold.

    Returns WallClimbVLMConfirmation if the VLM says climbing (for emitting alert), else None.
    """
    track_id = analysis.track_id

    # Use cache if we have a recent result
    if track_id in state.vlm_cache:
        cached = state.vlm_cache[track_id]
        if (analysis.timestamp - cached.timestamp).total_seconds() < 5.0:
            print(f"[WallClimb VLM] 📦 call_vlm: using cache track_id={track_id} climbing_detected={cached.climbing_detected}")
            return cached if cached.climbing_detected else None

    state.last_vlm_call_time[track_id] = analysis.timestamp.timestamp()

    prompt = build_wall_climb_detection_prompt()
    print(f"[WallClimb VLM] 📤 Sending 3 frames to VLM (track_id={track_id} person_index={analysis.person_index})")

    try:
        # Optional: save frames to disk for debugging
        for idx, frame in enumerate(frames):
            temp = WallClimbAnalysis(
                track_id=analysis.track_id,
                person_index=analysis.person_index,
                box=analysis.box,
                is_above_zone=analysis.is_above_zone,
                confidence=analysis.confidence,
                timestamp=analysis.timestamp,
                frame_index=analysis.frame_index + idx - len(frames) + 1,
            )
            frame_path = save_vlm_frame(frame, temp, frames_dir)
            print(f"[WallClimb VLM] 💾 Saved frame {idx + 1}/3 to {frame_path}")

        # Call Groq VLM with 3 images (crop to person box)
        vlm_result = vlm_service.analyze_images(
            frames,
            prompt=prompt,
            crop_box=analysis.box,
            temperature=0.1,
            max_tokens=500,
        )

        # Parse response (VLM may return a single object or a list of objects)
        raw_content = vlm_result.get("content") or ""
        print(f"[WallClimb VLM] 📥 VLM raw response (first 200 chars): {raw_content[:200]!r}")
        parsed = vlm_result.get("parsed_json")
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if parsed and isinstance(parsed, dict):
            climbing_detected = bool(parsed.get("climbing_detected", False))
            confidence = float(parsed.get("confidence", 0.0))
            description = str(parsed.get("description", ""))
            print(f"[WallClimb VLM] 📥 VLM parsed: climbing_detected={climbing_detected} confidence={confidence} description={description[:80]!r}...")
        else:
            content_lower = raw_content.lower()
            climbing_detected = any(
                w in content_lower for w in ["climbing", "climb", "jumping", "jump", "over", "wall", "fence"]
            )
            confidence = 0.5 if climbing_detected else 0.0
            description = raw_content
            print(f"[WallClimb VLM] 📥 VLM fallback parse: climbing_detected={climbing_detected} confidence={confidence} (no JSON)")

        confidence = max(0.0, min(1.0, confidence))
        confirmed = climbing_detected and confidence >= vlm_confidence_threshold
        print(f"[WallClimb VLM] 📥 VLM final: confirmed={confirmed} (threshold={vlm_confidence_threshold})")

        confirmation = WallClimbVLMConfirmation(
            track_id=analysis.track_id,
            person_index=analysis.person_index,
            box=analysis.box,
            climbing_detected=confirmed,
            confidence=confidence,
            description=description,
            vlm_response=vlm_result.get("raw_response", {}),
            timestamp=analysis.timestamp,
            frame_index=analysis.frame_index,
        )

        state.vlm_cache[track_id] = confirmation
        return confirmation if confirmed else None

    except Exception as e:
        print(f"[WallClimb VLM] ⚠️ VLM error: {e}")
        return None
