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
from app.processing.vision_tasks.tasks.fall_detection.vlm_handler import expand_box_for_vlm
from app.infrastructure.external.groq_vlm_service import GroqVLMService


def build_wall_climb_detection_prompt() -> str:
    """
    Prompt for the VLM: we send 3 images (before, suspicious moment, after).
    Ask for JSON with climbing_detected, confidence, description.
    Designed for accurate detection of wall-climb violation: trying, attempting, above wall, or any breach intent.
    """
    return """You are a security analyst. These 3 images are sequential video frames from a security camera: (1) BEFORE, (2) SUSPICIOUS MOMENT, (3) AFTER. Your task is to determine whether the person is committing or attempting a WALL-CLIMB / BARRIER-BREACH violation.

---
CRITICAL — NORMAL BEHAVIOUR (climbing_detected = false)
---
- If the person is SIMPLY STANDING near the wall or WALKING past/along the wall — this is NORMAL behaviour. Do NOT set climbing_detected = true. Only set true when the person actually TRIES TO CLIMB (reaching, gripping, pulling over, jumping over, or body above the barrier in a climbing action).
- Standing still next to the wall, walking alongside it, or leaning casually = NOT a violation. Set climbing_detected = false.

---
CONTEXT
---
- "Wall" means any vertical barrier: wall, fence, railing, gate, barrier, or similar structure meant to restrict access.
- A VIOLATION is: trying to climb, actively climbing, already above the wall, jumping over, or any behavior that shows intent to cross the barrier unlawfully.

---
WHAT TO DETECT AS VIOLATION (climbing_detected = true)
---

1) TRYING / PREPARING TO CLIMB
   - Hands reaching upward toward the top of the barrier (grasping motion or extended toward top).
   - Crouching or legs bent in a "ready to jump or push" posture next to the barrier.
   - Body oriented toward the barrier with arms raised or extended to grab.
   - Testing the barrier: touching, shaking, or feeling the top/surface as if assessing stability or height.
   - Standing very close to barrier with non-casual posture suggesting next step is to climb.

2) ACTIVELY CLIMBING
   - Upper body (chest, shoulders, head) above the top of the barrier.
   - Arms in pulling/gripping posture (elbows bent, hands on or over the barrier).
   - Legs in pushing or stepping-up motion (one leg raised, foot on barrier, or legs driving upward).
   - Body in a "scaling" or "pulling over" posture—clearly in the act of getting over.

3) PERSON ABOVE THE WALL
   - Torso or full body visibly above the barrier line (sitting on top, straddling, or already over).
   - Person in mid-crossing: one side over, other side still on original side.
   - Person descending the other side after having crossed (clear evidence they went over).

4) JUMPING OVER
   - Person airborne with body clearing the height of the barrier.
   - Legs or body in mid-air over the barrier in a jump/vault motion.

5) OTHER VIOLATION BEHAVIOUR
   - Any clear intent to circumvent the barrier: repeated attempts, stepping on lower parts to reach top, using objects to assist climbing, or body language that unmistakably indicates climbing/crossing intent.

---
WHAT IS NOT A VIOLATION (climbing_detected = false)
---
- Simply STANDING near the wall or WALKING past/along the wall — normal behaviour; never set climbing_detected = true for this alone.
- Walking or standing next to the wall with normal, upright posture (no reach, no grip, no climb posture).
- Leaning against the wall casually (back or shoulder against it, relaxed).
- Merely looking at or inspecting the barrier without climbing posture or hand/arm movement toward climbing.
- Body entirely below the barrier with arms at sides or in normal walking position—no upward movement or intent.
- Touching the wall briefly (e.g. hand on wall while walking) without grip or pull motion.
- Person on the same side throughout all 3 frames with no progression toward climbing.
- Until the person actually tries to climb (reach, grip, pull over, jump over), treat as normal — climbing_detected = false.

---
HOW TO ANALYSE THE 3 FRAMES
---
- Compare frame 1 → 2 → 3 for PROGRESSION: e.g. approach → reach/grip → pull over; or crouch → jump → over.
- Estimate where the barrier top is (even if not fully visible) and judge: is the person’s head, chest, or torso above that line?
- Focus on: arm position (reaching, gripping, pulling), leg position (bent to jump, stepping up, pushing), and body orientation relative to the barrier.
- One strong frame (e.g. clearly above wall or clear climbing posture) can be enough; support with other frames if present.
- If in doubt between "casual contact" and "climbing intent", prefer evidence: clear grip, clear above-barrier position, or clear jump = violation.

---
CONFIDENCE SCALE
---
- 0.90–1.00: Obvious violation — person clearly above wall, or clearly climbing/jumping over in at least one frame.
- 0.70–0.89: Strong evidence — clear preparation (hands up, grip, crouch to jump) or clear attempt (pulling up, one leg over).
- 0.50–0.69: Moderate evidence — posture or position suggests climbing intent but not definitive (e.g. hands near top, ambiguous).
- 0.30–0.49: Weak/ambiguous — could be casual; do not set climbing_detected = true unless you still see some intent.
- 0.00–0.29: No violation — normal standing, walking, or leaning; no climbing intent.

---
OUTPUT FORMAT
---
Respond ONLY with valid JSON, no other text:
{
    "climbing_detected": true or false,
    "confidence": 0.0 to 1.0,
    "description": "Concise description: (1) position of person relative to barrier in each relevant frame, (2) posture and arm/leg positions, (3) movement or progression across frames, (4) which violation type if any: trying/preparing, actively climbing, above wall, jumping over, or other; (5) one-sentence reasoning for your decision."
}

Be accurate: set climbing_detected = true only when there is clear evidence of trying to climb, actively climbing, person above the wall, jumping over, or other definite violation behaviour. When evidence is weak or ambiguous, set climbing_detected = false and use the lower end of the confidence range."""


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
    box_override: Optional[List[float]] = None,
) -> str:
    """Save one frame to disk (optional, for debugging). Uses box_override if provided, else analysis.box. Returns path."""
    os.makedirs(frames_dir, exist_ok=True)
    frame_filename = f"wall_climb_frame_{analysis.frame_index}_track_{analysis.track_id}_{int(analysis.timestamp.timestamp())}.jpg"
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
        # Expand box so VLM sees person + surroundings (reuse fall_detection helper)
        h, w = frames[0].shape[:2]
        expanded_box = expand_box_for_vlm(analysis.box, w, h)

        # Optional: save frames to disk for debugging (same crop as sent to VLM)
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
            frame_path = save_vlm_frame(frame, temp, frames_dir, box_override=expanded_box)
            print(f"[WallClimb VLM] 💾 Saved frame {idx + 1}/3 to {frame_path}")

        # Call Groq VLM with 3 images (crop to expanded box for context)
        vlm_result = vlm_service.analyze_images(
            frames,
            prompt=prompt,
            crop_box=expanded_box,
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
