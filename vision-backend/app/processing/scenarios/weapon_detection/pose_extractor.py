"""
Pose Extractor
--------------

Extracts pose keypoints and person bounding boxes from detection packets.
"""

from typing import Optional

from app.processing.scenarios.contracts import ScenarioFrameContext
from app.processing.scenarios.weapon_detection.types import PoseFrame


def extract_pose_frame(frame_context: ScenarioFrameContext) -> Optional[PoseFrame]:
    """
    Extract pose data from frame context.
    
    Filters detections to only "person" class and extracts:
    - Pose keypoints (if available)
    - Person bounding boxes
    
    Args:
        frame_context: Frame context with detections
        
    Returns:
        PoseFrame object if pose data available, None otherwise
    """
    detections = frame_context.detections
    
    # Check if we have keypoints (pose data)
    if not detections.keypoints or len(detections.keypoints) == 0:
        return None
    
    # Filter to only "person" class detections with keypoints
    # Keypoints are aligned with detections by index
    person_boxes = []
    person_keypoints = []
    
    for idx, detected_class in enumerate(detections.classes):
        if isinstance(detected_class, str) and detected_class.lower() == "person":
            # Check if we have a box and keypoints for this detection
            if idx < len(detections.boxes) and idx < len(detections.keypoints):
                keypoint_data = detections.keypoints[idx]
                # Only include if keypoints are not empty
                if keypoint_data and len(keypoint_data) > 0:
                    person_boxes.append(detections.boxes[idx])
                    person_keypoints.append(keypoint_data)
    
    # If no person detections with keypoints, return None
    if not person_keypoints or len(person_keypoints) == 0:
        return None
    
    # Create PoseFrame
    pose_frame = PoseFrame(
        frame=frame_context.frame,
        keypoints=person_keypoints,
        person_boxes=person_boxes,
        timestamp=frame_context.timestamp,
        frame_index=frame_context.frame_index
    )
    
    return pose_frame
