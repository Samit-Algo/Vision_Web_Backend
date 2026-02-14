"""
Face detection scenario
------------------------

Loads embeddings from person_gallery; runs face detection + recognition per frame.
Alerts when a watched person is identified. Uses DeepFace; matching by cosine similarity.
"""
# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent,
    OverlayData,
)
from app.processing.vision_tasks.task_lookup import register_scenario
from app.processing.vision_tasks.tasks.face_detection.config import FaceDetectionConfig
from app.utils.db import get_collection
from app.utils.face_embedding import (
    get_face_embeddings_from_frame_multi_detector,
    find_best_match,
    DEFAULT_EMBEDDING_MODEL,
)


def load_gallery_embeddings() -> Tuple[List[List[float]], List[str]]:
    """
    Load all (embedding, name) from person_gallery. Only persons with status "active"
    and matching embedding_model (DeepFace) are used. Each person can have multiple
    embeddings; all are added for accurate recognition.
    """
    coll = get_collection("person_gallery")
    cursor = coll.find(
        {"status": "active", "embedding_model": DEFAULT_EMBEDDING_MODEL},
        projection={"name": 1, "embeddings": 1},
    )
    known_encodings: List[List[float]] = []
    known_names: List[str] = []
    for doc in cursor:
        name = doc.get("name")
        embeddings = doc.get("embeddings")
        if name is None or not embeddings:
            continue
        name_str = str(name).strip()
        for emb in embeddings:
            if isinstance(emb, list) and len(emb) > 0:
                known_encodings.append(emb)
                known_names.append(name_str)
    return known_encodings, known_names


@register_scenario("face_detection")
class FaceDetectionScenario(BaseScenario):
    """
    Identifies persons in the frame using reference photos from person_gallery.
    When watch_names is set (e.g. ["sachin"]), alerts only when one of those persons appears.
    Emits person_identified events and stores recognized_faces in state for overlay.
    """

    def __init__(self, config: Dict[str, Any], pipeline_context: Any):
        super().__init__(config, pipeline_context)
        self.config_obj = FaceDetectionConfig(config, pipeline_context.task)
        self._state["last_alert_time"] = None
        self._state["recognized_faces"] = []  # [{ "box": [x1,y1,x2,y2], "name": "..." }, ...] - current frame detections only
        self._state["last_recognized_faces"] = []  # kept for state tracking (not used for overlay)
        self._state["last_recognized_time"] = None  # datetime of last non-empty recognition
        self._state["consecutive_frames_without_face"] = 0  # track consecutive frames without face (for state management)
        self._known_encodings: List[List[float]] = []
        self._known_names: List[str] = []
        self.load_gallery()

    def load_gallery(self) -> None:
        """Load all gallery embeddings from MongoDB."""
        try:
            self._known_encodings, self._known_names = load_gallery_embeddings()
            names_str = ", ".join(sorted(set(self._known_names))) if self._known_names else "none"
            print(f"[FaceDetection] ✅ Gallery loaded: {len(self._known_encodings)} encodings for names: [{names_str}] (min_similarity={self.config_obj.min_similarity})")
        except Exception as e:
            print(f"[FaceDetection] ⚠️ Failed to load gallery: {e}")
            self._known_encodings, self._known_names = [], []

    def requires_yolo_detections(self) -> bool:
        """Face detection uses raw frame only."""
        return False

    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        if not self._known_encodings:
            self._state["recognized_faces"] = []
            return []

        frame = frame_context.frame
        try:
            import cv2
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = frame
            small = frame

        # Use multi-detector (retinaface → mtcnn → opencv) for more consistent detection across frames
        face_boxes_embeddings = get_face_embeddings_from_frame_multi_detector(
            rgb,
            model_name=DEFAULT_EMBEDDING_MODEL,
            enforce_detection=True,
        )

        # Diagnostic: log every 60 frames so we can see if faces are detected and matched
        if frame_context.frame_index % 60 == 0:
            print(f"[FaceDetection] frame {frame_context.frame_index}: detected {len(face_boxes_embeddings)} face(s)")

        scale = 2.0 if frame.shape != small.shape else 1.0
        recognized: List[Dict[str, Any]] = []
        for (box_s, embedding) in face_boxes_embeddings:
            name = find_best_match(
                embedding,
                self._known_encodings,
                self._known_names,
                min_similarity=self.config_obj.min_similarity,
            ) or "Unknown"
            # Scale box back to full frame coordinates [left, top, right, bottom]
            box = [
                float(box_s[0] * scale),
                float(box_s[1] * scale),
                float(box_s[2] * scale),
                float(box_s[3] * scale),
            ]
            recognized.append({"box": box, "name": name})

        self._state["recognized_faces"] = recognized
        if recognized:
            self._state["last_recognized_faces"] = recognized
            self._state["last_recognized_time"] = frame_context.timestamp
            self._state["consecutive_frames_without_face"] = 0
        else:
            self._state["consecutive_frames_without_face"] = self._state.get("consecutive_frames_without_face", 0) + 1
            # After many consecutive frames with no face, clear saved state (for debugging/logging)
            if self._state["consecutive_frames_without_face"] >= 15:
                self._state["last_recognized_faces"] = []
                self._state["last_recognized_time"] = None

        if frame_context.frame_index % 60 == 0 and recognized:
            names_in_frame = [r.get("name", "Unknown") for r in recognized]
            print(f"[FaceDetection] frame {frame_context.frame_index}: recognized names = {names_in_frame}")

        # Emit alert only for watched names (or any recognized if watch_names empty)
        watch_set = set(self.config_obj.watch_names) if self.config_obj.watch_names else None
        alerted_names: List[str] = []
        for r in recognized:
            n = (r.get("name") or "").strip().lower()
            if n and n != "unknown":
                if watch_set is None or n in watch_set:
                    alerted_names.append(r.get("name", ""))

        if not alerted_names:
            return []

        now_ts = frame_context.timestamp
        last_alert = self._state.get("last_alert_time")
        if last_alert and isinstance(last_alert, datetime):
            if (now_ts - last_alert).total_seconds() < self.config_obj.alert_cooldown_seconds:
                if frame_context.frame_index % 60 == 0:
                    remaining = self.config_obj.alert_cooldown_seconds - (now_ts - last_alert).total_seconds()
                    print(f"[FaceDetection] Would alert for {alerted_names} but cooldown (next in {remaining:.1f}s)")
                return []

        self._state["last_alert_time"] = now_ts
        label = self.config_obj.custom_label or f"{', '.join(alerted_names)} detected"
        return [
            ScenarioEvent(
                event_type="person_identified",
                label=label,
                confidence=1.0,
                metadata={
                    "names": alerted_names,
                    "recognized_faces": recognized,
                },
                detection_indices=[],  # No YOLO indices for face
                timestamp=now_ts,
                frame_index=frame_context.frame_index,
            )
        ]

    def get_overlay_data(self, frame_context: Optional[ScenarioFrameContext] = None) -> List[Any]:
        """Return boxes and names for recognized faces (for UI overlay).
        Box shows ONLY when person is detected in current frame.
        If no face detected in current frame, no box is shown.
        """
        # Only use current frame detections - no smoothing or last saved faces
        faces = self._state.get("recognized_faces") or []
        out: List[OverlayData] = []
        for f in faces:
            box = f.get("box")
            name = f.get("name") or "Unknown"
            if box and len(box) >= 4:
                out.append(
                    OverlayData(
                        box=box,
                        label=name,
                        color=(0, 255, 102),  # Green
                    )
                )
        return out

    def reset(self) -> None:
        self._state["last_alert_time"] = None
        self._state["recognized_faces"] = []
        self._state["last_recognized_faces"] = []
        self._state["last_recognized_time"] = None
        self._state["consecutive_frames_without_face"] = 0
        self.load_gallery()
