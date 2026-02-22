"""
Sleep Detection Scenario
-------------------------
Pose → buffer → sleep posture analysis (lying/head-down) → VLM confirmation → person_sleeping event.
Only emits when VLM confirms. Pipeline uses sleep_confirmed_indices for red boxes.

Code layout:
  - box_iou: IoU of two boxes (for matching).
  - SleepDetectionScenario: __init__, process (extract pose → buffer → deferred VLM → events), get_overlay_data.
"""

# -------- Imports --------
import os
from typing import Any, Dict, List, Optional

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
    OverlayData,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from app.processing.vision_tasks.tasks.sleep_detection.config import SleepDetectionConfig
from app.processing.vision_tasks.tasks.sleep_detection.state import SleepDetectionState
from app.processing.vision_tasks.tasks.sleep_detection.pose_extractor import extract_pose_frame
from app.processing.vision_tasks.tasks.sleep_detection.types import SleepAnalysis
from app.processing.vision_tasks.tasks.sleep_detection.sleep_analyzer import analyze_sleep_posture
from app.processing.vision_tasks.tasks.sleep_detection.vlm_handler import (
    should_call_vlm,
    call_vlm,
    get_person_key,
)
from app.infrastructure.external.groq_vlm_service import GroqVLMService


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Intersection-over-union of two boxes [x1, y1, x2, y2]. Returns 0 if no overlap."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ========== Scenario: Sleep detection (pose → buffer → VLM confirm) ==========

@register_scenario("sleep_detection")
class SleepDetectionScenario(BaseScenario):
    """
    Sleep detection: pose → temporal consistency → VLM confirmation.

    We only emit an event when the VLM confirms the person is sleeping
    (so we avoid false alerts from bending, looking down, etc.).
    """

    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        self.config_obj = SleepDetectionConfig(config)
        self.state = SleepDetectionState(self.config_obj.buffer_size)
        self._vlm_service: Optional[GroqVLMService] = None
        os.makedirs(self.config_obj.vlm_frames_dir, exist_ok=True)

    def get_vlm_service(self) -> GroqVLMService:
        """Create VLM service once (lazy)."""
        if self._vlm_service is None:
            self._vlm_service = GroqVLMService()
        return self._vlm_service

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process one frame. Returns a list of events (usually 0 or 1: person_sleeping).
        """
        events = []
        pose_frame = extract_pose_frame(frame_context)
        if not pose_frame:
            return events
        self.state.add_pose_frame(pose_frame)
        print(f"[SleepDetection Scenario] 📦 Buffer size: {len(self.state.pose_buffer)} (frame_index={frame_context.frame_index})")

        vlm_service = self.get_vlm_service()
        # --- Process deferred VLM when we have 3 frames ---
        to_remove = []
        for analysis, suspicious_frame_index in list(self.state.deferred_vlm):
            if frame_context.frame_index != suspicious_frame_index + 1:
                continue
            if len(self.state.pose_buffer) < 3:
                continue
            before_frame = self.state.pose_buffer[-3].frame
            suspicious_frame = self.state.pose_buffer[-2].frame
            after_frame = self.state.pose_buffer[-1].frame
            three_frames = [before_frame, suspicious_frame, after_frame]

            print(f"[SleepDetection Scenario] 📤 Calling VLM for person_index={analysis.person_index} (suspicious_frame={suspicious_frame_index}, 3 frames)")
            vlm_result = call_vlm(
                analysis,
                three_frames,
                self.state,
                vlm_service,
                self.config_obj.vlm_confidence_threshold,
                self.config_obj.vlm_frames_dir,
            )
            to_remove.append((analysis, suspicious_frame_index))

            if vlm_result:
                print(f"[SleepDetection Scenario] 📥 VLM response: sleeping_detected={vlm_result.sleeping_detected} confidence={vlm_result.confidence:.2f} description={vlm_result.description[:80]!r}...")
            else:
                print(f"[SleepDetection Scenario] 📥 VLM response: no confirmation (None or not sleeping)")

            if vlm_result and vlm_result.sleeping_detected:
                person_key = get_person_key(analysis.person_index, analysis.box)
                last_emitted = self.state.emitted_events.get(person_key)
                if last_emitted is None or (frame_context.timestamp - last_emitted).total_seconds() > 5.0:
                    print(f"[SleepDetection Scenario] 🚨 EVENT: person_sleeping emitted (person_index={analysis.person_index} confidence={vlm_result.confidence:.2f})")
                    events.append(
                        ScenarioEvent(
                            event_type="person_sleeping",
                            label=f"Person sleeping (confidence: {vlm_result.confidence:.2f})",
                            confidence=vlm_result.confidence,
                            metadata={
                                "person_index": vlm_result.person_index,
                                "box": vlm_result.box,
                                "reason": analysis.reason,
                                "vlm_description": vlm_result.description,
                                "vlm_response": vlm_result.vlm_response,
                            },
                            detection_indices=[vlm_result.person_index],
                            timestamp=vlm_result.timestamp,
                            frame_index=vlm_result.frame_index,
                        )
                    )
                    self.state.emitted_events[person_key] = frame_context.timestamp
                    self.state.emitted_event_boxes[person_key] = list(vlm_result.box)

        for item in to_remove:
            self.state.deferred_vlm.remove(item)

        # Step 4: If buffer is full enough, run sleep posture analysis
        current_analyses: List[SleepAnalysis] = []
        if len(self.state.pose_buffer) >= self.config_obj.temporal_consistency_frames:
            analyses = analyze_sleep_posture(
                self.state.pose_buffer,
                self.config_obj.temporal_consistency_frames,
                self.config_obj.torso_angle_lying_deg,
                self.config_obj.head_down_angle_deg,
                self.config_obj.motion_threshold_px,
                self.config_obj.kp_confidence_threshold,
                getattr(self.config_obj, "min_nose_below_shoulder_px", 10.0),
                getattr(self.config_obj, "head_down_majority_ratio", 0.5),
            )
            current_analyses = analyses
            if analyses:
                print(f"[SleepDetection Scenario] 🔍 Sleep posture analysis: {len(analyses)} possibly sleeping (reasons: {[a.reason for a in analyses]})")
            self.state.pending_analyses.extend(analyses)

        # Clear red "SLEEPING CONFIRMED" when same person is no longer possibly sleeping (woke / moved)
        if len(self.state.pose_buffer) > 0:
            current_boxes = self.state.pose_buffer[-1].person_boxes
            possibly_sleeping_boxes = [a.box for a in current_analyses]
            for person_key in list(self.state.emitted_event_boxes.keys()):
                emitted_box = self.state.emitted_event_boxes[person_key]
                for curr_box in current_boxes:
                    if box_iou(emitted_box, curr_box) < 0.3:
                        continue
                    # Same person (overlap). Is they in the "possibly sleeping" set?
                    if not any(box_iou(curr_box, b) >= 0.3 for b in possibly_sleeping_boxes):
                        self.state.emitted_events.pop(person_key, None)
                        self.state.emitted_event_boxes.pop(person_key, None)
                    break

        # Update "stable since" (5s still before VLM): only keep persons who are possibly_sleeping AND is_still
        still_and_possibly = [(get_person_key(a.person_index, a.box), a.box, a.timestamp) for a in current_analyses if a.possibly_sleeping and getattr(a, "is_still", True)]
        for person_key, box, ts in still_and_possibly:
            # Same person (by box IoU)? Keep earliest timestamp
            found = False
            for key, (stable_ts, stable_box) in list(self.state.person_stable_since.items()):
                if box_iou(box, stable_box) >= 0.3:
                    found = True
                    if key != person_key:
                        self.state.person_stable_since.pop(key, None)
                        self.state.person_stable_since[person_key] = (stable_ts, box)
                    break
            if not found:
                self.state.person_stable_since[person_key] = (ts, list(box))
        # Remove entries for persons no longer still+possibly_sleeping
        for key in list(self.state.person_stable_since.keys()):
            _, stable_box = self.state.person_stable_since[key]
            if not any(box_iou(stable_box, b) >= 0.3 for _, b, _ in still_and_possibly):
                self.state.person_stable_since.pop(key, None)

        # Step 5: Defer VLM only after 5 seconds of no movement (possibly_sleeping + is_still)
        trigger_seconds = getattr(self.config_obj, "vlm_trigger_still_seconds", 5.0)
        deferred_keys = {get_person_key(a.person_index, a.box) for a, _ in self.state.deferred_vlm}
        now_ts = frame_context.timestamp
        for analysis in list(self.state.pending_analyses):
            if not analysis.possibly_sleeping:
                continue
            person_key = get_person_key(analysis.person_index, analysis.box)
            if person_key in deferred_keys:
                continue
            # Find this person's stable_since (by box overlap)
            stable_ts = None
            for key, (ts, box) in self.state.person_stable_since.items():
                if box_iou(analysis.box, box) >= 0.3:
                    stable_ts = ts
                    break
            if stable_ts is None:
                continue
            elapsed = (now_ts - stable_ts).total_seconds()
            if elapsed < trigger_seconds:
                print(f"[SleepDetection Scenario] ⏳ Waiting {trigger_seconds - elapsed:.1f}s still before VLM (person_key={person_key[:30]}...)")
                continue
            if not should_call_vlm(analysis, self.state, self.config_obj.vlm_throttle_seconds):
                print(f"[SleepDetection Scenario] ⏳ VLM throttled/skip for person_key={person_key}")
                continue
            self.state.deferred_vlm.append((analysis, analysis.frame_index))
            deferred_keys.add(person_key)
            print(f"[SleepDetection Scenario] 📋 Deferred VLM for person_index={analysis.person_index} reason={analysis.reason} (still {elapsed:.1f}s, will call on next frame)")

        # Step 6: Clean up old data
        self.state.cleanup_old_data(frame_context.timestamp)

        return events

    def reset(self) -> None:
        """Clear all state when rule is disabled."""
        self.state.reset()
        self._vlm_service = None

    def requires_yolo_detections(self) -> bool:
        """We need YOLO pose (person + keypoints)."""
        return True

    def get_overlay_data(self, frame_context: Optional[ScenarioFrameContext] = None) -> List[OverlayData]:
        """No separate overlay: pipeline uses emitted_event_boxes to set sleep_confirmed_indices so the same person box turns red."""
        return []
