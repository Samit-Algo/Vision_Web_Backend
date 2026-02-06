"""
VLM Handler
-----------

Step 3: Send 3 frames to the VLM and ask: "Is this person sleeping?"

We only call the VLM when pose analysis says "possibly sleeping" (to save cost).
We send: frame before, frame when we flagged, frame after.
We parse the VLM reply: sleeping_detected (true/false), confidence, description.
"""

import os
from typing import Optional, List
import cv2
import numpy as np

from app.processing.vision_tasks.tasks.sleep_detection.types import (
    SleepAnalysis,
    SleepVLMConfirmation,
)
from app.processing.vision_tasks.tasks.sleep_detection.state import SleepDetectionState
from app.infrastructure.external.groq_vlm_service import GroqVLMService


def build_sleep_detection_prompt() -> str:
    """
    Prompt for the VLM: we send 3 images (before, suspicious moment, after).
    Ask for JSON with sleeping_detected, confidence, description.
    """
    return """These 3 images are sequential video frames from a factory/industrial camera: (1) before, (2) the moment we suspect the person may be sleeping, (3) after.

Determine if the person visible in the frames is SLEEPING (e.g. lying down asleep, or standing but asleep/nodding off, eyes closed or head down and still). If the person is clearly awake (working, walking, looking at something), say not sleeping.

Respond ONLY with a valid JSON object in this exact format:
{
    "sleeping_detected": true or false,
    "confidence": 0.0 to 1.0,
    "description": "Brief description of what you see (posture, eyes if visible, activity)"
}

Be strict: only set sleeping_detected to true if you are confident the person is actually sleeping. False positives (alerting when the person is awake) are worse than false negatives."""


def get_person_key(person_index: int, box: List[float]) -> str:
    """Unique key for this person (for caching and throttling)."""
    box_str = ",".join(f"{v:.1f}" for v in box[:4])
    return f"person_{person_index}_{box_str}"


def should_call_vlm(
    analysis: SleepAnalysis,
    state: SleepDetectionState,
    vlm_throttle_seconds: float,
) -> bool:
    """
    Decide if we should call the VLM for this person now.
    - Don't call if we have a recent cached result (within 5 seconds).
    - Don't call if we called VLM for this person too recently (throttle).
    """
    person_key = get_person_key(analysis.person_index, analysis.box)

    if person_key in state.vlm_cache:
        cached = state.vlm_cache[person_key]
        if (analysis.timestamp - cached.timestamp).total_seconds() < 5.0:
            print(f"[SleepDetection VLM] ⏭️ should_call_vlm: skip (cached recent) person_key={person_key}")
            return False

    last_call = state.last_vlm_call_time.get(person_key, 0.0)
    current_time = analysis.timestamp.timestamp()
    if current_time - last_call < vlm_throttle_seconds:
        print(f"[SleepDetection VLM] ⏭️ should_call_vlm: skip (throttle) person_key={person_key} last_call_ago={current_time - last_call:.1f}s")
        return False

    print(f"[SleepDetection VLM] ✅ should_call_vlm: yes person_key={person_key}")
    return True


def save_vlm_frame(
    frame: np.ndarray,
    analysis: SleepAnalysis,
    frames_dir: str,
) -> str:
    """Save one frame to disk (optional, for debugging). Returns path."""
    os.makedirs(frames_dir, exist_ok=True)
    frame_filename = f"sleep_frame_{analysis.frame_index}_person_{analysis.person_index}_{int(analysis.timestamp.timestamp())}.jpg"
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
    analysis: SleepAnalysis,
    frames: List[np.ndarray],
    state: SleepDetectionState,
    vlm_service: GroqVLMService,
    vlm_confidence_threshold: float,
    frames_dir: str,
) -> Optional[SleepVLMConfirmation]:
    """
    Send the 3 frames to the VLM and parse the response.

    - frames: [before_frame, suspicious_frame, after_frame]
    - We crop to the person's box so the VLM sees the person clearly.
    - We parse JSON: sleeping_detected, confidence, description.
    - We only "confirm" sleep if sleeping_detected is true and confidence >= threshold.

    Returns SleepVLMConfirmation if the VLM says sleeping (for emitting alert), else None.
    """
    person_key = get_person_key(analysis.person_index, analysis.box)

    # Use cache if we have a recent result
    if person_key in state.vlm_cache:
        cached = state.vlm_cache[person_key]
        if (analysis.timestamp - cached.timestamp).total_seconds() < 5.0:
            print(f"[SleepDetection VLM] 📦 call_vlm: using cache person_key={person_key} sleeping_detected={cached.sleeping_detected}")
            return cached if cached.sleeping_detected else None

    state.last_vlm_call_time[person_key] = analysis.timestamp.timestamp()

    prompt = build_sleep_detection_prompt()
    print(f"[SleepDetection VLM] 📤 Sending 3 frames to VLM (person_index={analysis.person_index} reason={analysis.reason})")

    try:
        # Optional: save frames to disk for debugging
        for idx, frame in enumerate(frames):
            temp = SleepAnalysis(
                person_index=analysis.person_index,
                box=analysis.box,
                possibly_sleeping=analysis.possibly_sleeping,
                reason=analysis.reason,
                confidence=analysis.confidence,
                is_still=getattr(analysis, "is_still", True),
                timestamp=analysis.timestamp,
                frame_index=analysis.frame_index + idx - len(frames) + 1,
            )
            frame_path = save_vlm_frame(frame, temp, frames_dir)
            print(f"[SleepDetection VLM] 💾 Saved frame {idx + 1}/3 to {frame_path}")

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
        print(f"[SleepDetection VLM] 📥 VLM raw response (first 200 chars): {raw_content[:200]!r}")
        parsed = vlm_result.get("parsed_json")
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if parsed and isinstance(parsed, dict):
            sleeping_detected = bool(parsed.get("sleeping_detected", False))
            confidence = float(parsed.get("confidence", 0.0))
            description = str(parsed.get("description", ""))
            print(f"[SleepDetection VLM] 📥 VLM parsed: sleeping_detected={sleeping_detected} confidence={confidence} description={description[:80]!r}...")
        else:
            content_lower = raw_content.lower()
            sleeping_detected = any(
                w in content_lower for w in ["sleeping", "asleep", "sleep", "nodding"]
            )
            confidence = 0.5 if sleeping_detected else 0.0
            description = raw_content
            print(f"[SleepDetection VLM] 📥 VLM fallback parse: sleeping_detected={sleeping_detected} confidence={confidence} (no JSON)")

        confidence = max(0.0, min(1.0, confidence))
        confirmed = sleeping_detected and confidence >= vlm_confidence_threshold
        print(f"[SleepDetection VLM] 📥 VLM final: confirmed={confirmed} (threshold={vlm_confidence_threshold})")

        confirmation = SleepVLMConfirmation(
            person_index=analysis.person_index,
            box=analysis.box,
            sleeping_detected=confirmed,
            confidence=confidence,
            description=description,
            vlm_response=vlm_result.get("raw_response", {}),
            timestamp=analysis.timestamp,
            frame_index=analysis.frame_index,
        )

        state.vlm_cache[person_key] = confirmation
        return confirmation if confirmed else None

    except Exception as e:
        print(f"[SleepDetection VLM] ⚠️ VLM error: {e}")
        return None
