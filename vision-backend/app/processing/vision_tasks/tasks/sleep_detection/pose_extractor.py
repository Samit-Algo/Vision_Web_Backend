"""
Pose Extractor
--------------

Step 1: Get pose data from the current frame.

We take the pipeline’s detections and:
- Keep only "person" detections that have keypoints (YOLO pose gives us keypoints).
- Build one PoseFrame with: frame image, list of keypoints per person, list of boxes.

If there are no persons with keypoints, we return None (next step will skip).
"""

from typing import Optional

from app.processing.vision_tasks.data_models import ScenarioFrameContext
from app.processing.vision_tasks.tasks.sleep_detection.types import PoseFrame


def extract_pose_frame(frame_context: ScenarioFrameContext) -> Optional[PoseFrame]:
    """
    Extract pose data from the frame context.

    - Filter detections to class "person" only.
    - Only include persons that have keypoints (from YOLO pose model).
    - Return a PoseFrame with frame, keypoints, boxes, timestamp, frame_index.

    Returns:
        PoseFrame if we have at least one person with keypoints, else None.
    """
    detections = frame_context.detections

    # Pipeline must give us keypoints (from YOLO pose)
    if not detections.keypoints or len(detections.keypoints) == 0:
        print(f"[SleepDetection PoseExtractor] ⏭️ No keypoints in frame (frame_index={frame_context.frame_index}) - skip")
        return None

    person_boxes = []
    person_keypoints = []

    # Loop over each detection; index matches boxes and keypoints
    for idx, detected_class in enumerate(detections.classes):
        if not (isinstance(detected_class, str) and detected_class.lower() == "person"):
            continue
        # Must have a box and keypoints for this index
        if idx >= len(detections.boxes) or idx >= len(detections.keypoints):
            continue
        keypoint_data = detections.keypoints[idx]
        if not keypoint_data or len(keypoint_data) == 0:
            continue
        person_boxes.append(detections.boxes[idx])
        person_keypoints.append(keypoint_data)

    if not person_keypoints:
        print(f"[SleepDetection PoseExtractor] ⏭️ No person with keypoints in frame (frame_index={frame_context.frame_index}) - skip")
        return None

    print(f"[SleepDetection PoseExtractor] ✅ Person detection: {len(person_keypoints)} person(s) with pose (frame_index={frame_context.frame_index})")
    return PoseFrame(
        frame=frame_context.frame,
        keypoints=person_keypoints,
        person_boxes=person_boxes,
        timestamp=frame_context.timestamp,
        frame_index=frame_context.frame_index,
    )
