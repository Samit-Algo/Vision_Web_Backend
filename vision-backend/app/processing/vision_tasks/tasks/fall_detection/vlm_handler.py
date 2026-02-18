"""
VLM Handler for Fall Detection
--------------------------------

Send 5 sequential frames to the VLM when fall is suspected (Groq API max 5 images).
We parse the VLM reply: fall_detected (true/false), confidence, description.
"""

import os
from typing import Optional, List
import cv2
import numpy as np

from app.processing.vision_tasks.tasks.fall_detection.types import (
    FallAnalysis,
    FallVLMConfirmation,
)
from app.processing.vision_tasks.tasks.fall_detection.state import FallDetectionState
from app.infrastructure.external.groq_vlm_service import GroqVLMService


def build_fall_detection_prompt() -> str:
    return """You are a fall-detection analyst. You see 5 sequential security-camera frames (Frame 1 = oldest, Frame 5 = newest) showing the same person. Your job is to decide: should we report a FALL (person may need help)?
is the person on the going to fall (make it true) 
---
STEP 1 — Is the person ON THE FLOOR or GROUND?
---
Look at the person's position in the frames, especially the last frames (4 and 5).

• If the person is LYING FLAT or HORIZONTAL on the ground (body stretched out, prone or on back/side on a floor surface), they are "on the floor".
• "On the floor" means: body is on a floor-like surface (tiles, concrete, wood, laminate, indoor/outdoor hard surface, garage floor, porch, corridor, room floor). This is a POSSIBLE FALL — we need to alert so someone can check.

---
STEP 2 — What kind of surface are they on?
---
• HARD FLOOR (tiles, concrete, wood, garage, porch, corridor, room floor, pavement): If the person is lying on this → treat as FALL. Set fall_detected = true. They may have fallen and need help.
• BED, SOFA, MATTRESS, COUCH, PILLOW, YOGA MAT: If clearly on a soft/furniture surface → set fall_detected = false (intentional rest).
• If you cannot tell the surface but the person is horizontal and on a ground-level surface → prefer fall_detected = true (better to alert than miss a real fall).

---
STEP 3 — Motion in the sequence
---
• If you see a clear FALL: person was upright (standing/walking) and then goes down (tripping, slipping, losing balance, collapsing) → fall_detected = true.
• If the person is ALREADY on the floor in (almost) all frames: still set fall_detected = true if they are on a HARD FLOOR (tiles, concrete, garage, room floor, etc.). We did not see the moment of fall but they are on the ground and may need help.
• Only set false if: they are clearly on bed/sofa/mat, OR they are sitting/kneeling/crouching intentionally, OR they are standing/walking/getting up, OR they are exercising/playing/yoga.

---
DECISION RULE (follow this):
---
- Person lying on floor (tiles, concrete, garage, room floor, corridor, porch, any hard ground) → fall_detected = TRUE. Confidence 0.7–1.0.
- Person on bed/sofa/mattress/couch/mat → fall_detected = FALSE.
- Person standing, walking, sitting in a chair, kneeling, crouching, getting up → fall_detected = FALSE.
- Uncontrolled fall (tripping, slipping, collapse) visible in sequence → fall_detected = TRUE.

When in doubt and the person is horizontal on a floor-like surface → use true (we prefer to alert).

---
Respond ONLY with this JSON, no other text:
{"fall_detected": true or false, "confidence": 0.0 to 1.0, "description": "One short sentence: what you see and why."}"""


def should_call_vlm(
    analysis: FallAnalysis,
    state: FallDetectionState,
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
            print(f"[Fall Detection VLM] ⏭️ should_call_vlm: skip (cached recent) track_id={track_id}")
            return False

    last_call = state.last_vlm_call_time.get(track_id, 0.0)
    current_time = analysis.timestamp.timestamp()
    if current_time - last_call < vlm_throttle_seconds:
        print(f"[Fall Detection VLM] ⏭️ should_call_vlm: skip (throttle) track_id={track_id} last_call_ago={current_time - last_call:.1f}s")
        return False

    print(f"[Fall Detection VLM] ✅ should_call_vlm: yes track_id={track_id}")
    return True


def expand_box_for_vlm(
    box: List[float],
    img_width: int,
    img_height: int,
    margin_ratio: float = 0.35,
    min_padding_px: int = 45,
) -> List[float]:
    """Expand box by margin so VLM sees more context (person + surroundings). Returns [x1, y1, x2, y2] clamped to image."""
    x1, y1, x2, y2 = box[:4]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_w = max(bw * margin_ratio, min_padding_px)
    pad_h = max(bh * margin_ratio, min_padding_px)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img_width, x2 + pad_w)
    y2 = min(img_height, y2 + pad_h)
    return [float(x1), float(y1), float(x2), float(y2)]


def save_vlm_frame(
    frame: np.ndarray,
    analysis: FallAnalysis,
    frames_dir: str,
    frame_label: str = "",
    box_override: Optional[List[float]] = None,
) -> str:
    """Save one frame to disk (optional, for debugging). Uses box_override if provided, else analysis.box."""
    os.makedirs(frames_dir, exist_ok=True)
    frame_filename = f"fall_frame_{analysis.frame_index}_track_{analysis.track_id}_{int(analysis.timestamp.timestamp())}{frame_label}.jpg"
    frame_path = os.path.join(frames_dir, frame_filename)

    box = box_override if box_override is not None else analysis.box
    x1, y1, x2, y2 = [int(c) for c in box[:4]]
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
    analysis: FallAnalysis,
    frames: List[np.ndarray],
    state: FallDetectionState,
    vlm_service: GroqVLMService,
    vlm_confidence_threshold: float,
    frames_dir: str,
) -> Optional[FallVLMConfirmation]:
    """
    Send 5 sequential frames to the VLM and parse the response (Groq API max 5 images).
    - frames: 5 images (oldest to newest). We parse JSON: fall_detected, confidence, description.
    - Confirm fall only if fall_detected is true and confidence >= threshold.
    Returns FallVLMConfirmation if the VLM says fall (for emitting alert), else None.
    """
    track_id = analysis.track_id

    # Use cache if we have a recent result
    if track_id in state.vlm_cache:
        cached = state.vlm_cache[track_id]
        if (analysis.timestamp - cached.timestamp).total_seconds() < 5.0:
            print(f"[Fall Detection VLM] 📦 call_vlm: using cache track_id={track_id} fall_detected={cached.fall_detected}")
            return cached if cached.fall_detected else None

    state.last_vlm_call_time[track_id] = analysis.timestamp.timestamp()

    prompt = build_fall_detection_prompt()
    print(f"[Fall Detection VLM] 📤 Sending {len(frames)} frames to VLM (track_id={track_id} person_index={analysis.person_index})")
    
    if len(frames) != 5:
        print(f"[Fall Detection VLM] ⚠️ Warning: Expected 5 frames, got {len(frames)}")

    # Expand crop so VLM sees more context (person + surroundings) for better visibility
    h, w = frames[0].shape[:2]
    crop_box = expand_box_for_vlm(analysis.box, w, h, margin_ratio=0.2, min_padding_px=25)

    try:
        # Optional: save frames to disk for debugging (use expanded box)
        frame_labels = ["_1", "_2", "_3", "_4", "_5"]
        for idx, frame in enumerate(frames):
            label = frame_labels[idx] if idx < len(frame_labels) else f"_{idx+1}"
            frame_path = save_vlm_frame(frame, analysis, frames_dir, label, box_override=crop_box)
            print(f"[Fall Detection VLM] 💾 Saved frame {idx + 1}/{len(frames)} to {frame_path}")

        # Call Groq VLM with 5 images (API limit; expanded crop for better visibility)
        vlm_result = vlm_service.analyze_images(
            frames,
            prompt=prompt,
            crop_box=crop_box,
            temperature=0.1,
            max_tokens=800,
        )

        # Parse response (VLM may return a single object or a list of objects)
        raw_content = vlm_result.get("content") or ""
        print(f"[Fall Detection VLM] 📥 VLM raw response (first 300 chars): {raw_content[:300]!r}")
        parsed = vlm_result.get("parsed_json")
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if parsed and isinstance(parsed, dict):
            fall_detected = bool(parsed.get("fall_detected", False))
            confidence = float(parsed.get("confidence", 0.0))
            description = str(parsed.get("description", ""))
            print(f"[Fall Detection VLM] 📥 VLM parsed: fall_detected={fall_detected} confidence={confidence} description={description[:100]!r}...")
        else:
            # Fallback parsing if JSON not found
            content_lower = raw_content.lower()
            fall_detected = any(
                w in content_lower for w in ["fall", "fallen", "falling", "collapse", "collapsed", 
                                             "lost balance", "tripped", "slipped", "uncontrolled"]
            ) and not any(
                w in content_lower for w in ["no fall", "not fall", "false", "intentional", "sitting", "lying down"]
            )
            confidence = 0.5 if fall_detected else 0.0
            description = raw_content
            print(f"[Fall Detection VLM] 📥 VLM fallback parse: fall_detected={fall_detected} confidence={confidence} (no JSON)")

        confidence = max(0.0, min(1.0, confidence))
        confirmed = fall_detected and confidence >= vlm_confidence_threshold
        print(f"[Fall Detection VLM] 📥 VLM final: confirmed={confirmed} (threshold={vlm_confidence_threshold})")

        confirmation = FallVLMConfirmation(
            track_id=analysis.track_id,
            person_index=analysis.person_index,
            box=analysis.box,
            fall_detected=confirmed,
            confidence=confidence,
            description=description,
            vlm_response=vlm_result.get("raw_response", {}),
            timestamp=analysis.timestamp,
            frame_index=analysis.frame_index,
        )

        state.vlm_cache[track_id] = confirmation
        return confirmation if confirmed else None

    except Exception as e:
        print(f"[Fall Detection VLM] ⚠️ VLM error: {e}")
        import traceback
        traceback.print_exc()
        return None
