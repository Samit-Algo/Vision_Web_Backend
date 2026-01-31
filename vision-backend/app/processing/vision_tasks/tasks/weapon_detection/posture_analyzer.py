"""
Arm Posture Analyzer
--------------------

Analyzes arm posture from pose keypoints to detect suspicious patterns.
Uses the same person-pose logic as the old weapon_detection scenario:
- YOLO/COCO keypoint indices for shoulders, elbows, wrists
- Optional keypoint confidence threshold (only use reliable keypoints)
- Temporal consistency across buffered frames
"""

from typing import List, Optional, Tuple
import math

from app.processing.vision_tasks.tasks.weapon_detection.types import (
    PoseFrame,
    ArmPostureAnalysis
)


# YOLO/COCO pose keypoint indices (same as old code)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10


def _get_keypoint(
    person_keypoints: List[List[float]],
    idx: int,
    confidence_threshold: float = 0.0
) -> Optional[Tuple[float, float]]:
    """
    Get keypoint (x, y) if available and confident (old-code style, like fall_detection).
    
    Args:
        person_keypoints: List of keypoints for one person [[x, y] or [x, y, conf], ...]
        idx: Keypoint index (COCO: 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow, 9=left_wrist, 10=right_wrist)
        confidence_threshold: Minimum confidence to consider keypoint valid (0 = use all)
    
    Returns:
        (x, y) tuple if valid, None otherwise
    """
    if idx >= len(person_keypoints):
        return None
    
    kp = person_keypoints[idx]
    if not kp or len(kp) < 2:
        return None
    
    # Check confidence if available (3rd element)
    if len(kp) >= 3 and confidence_threshold > 0 and kp[2] < confidence_threshold:
        return None
    
    return float(kp[0]), float(kp[1])


def _wrist_or_elbow_at_shoulder_height(
    shoulder: Tuple[float, float],
    elbow: Optional[Tuple[float, float]],
    wrist: Tuple[float, float],
) -> bool:
    """
    Check if wrist or elbow is at or above shoulder height (Option A).
    Image Y: smaller = higher. So at-or-above means y <= shoulder_y.
    """
    sy = shoulder[1]
    if wrist[1] <= sy:
        return True
    if elbow is not None and elbow[1] <= sy:
        return True
    return False


def check_single_arm(
    keypoints: List[List[float]],
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
    arm_angle_threshold: float,
    kp_confidence_threshold: float = 0.0,
    require_shoulder_height: bool = True,
) -> tuple[bool, float]:
    """
    Check if a single arm is raised (angle + optional shoulder-height check).
    
    Uses shoulder -> wrist vector angle from horizontal. Option A: also requires
    wrist or elbow at or above shoulder height when require_shoulder_height is True.
    
    Args:
        keypoints: List of keypoints for one person
        shoulder_idx: Shoulder keypoint index
        elbow_idx: Elbow keypoint index
        wrist_idx: Wrist keypoint index
        arm_angle_threshold: Minimum angle (degrees) to consider arm raised
        kp_confidence_threshold: Min keypoint confidence (0 = use all)
        require_shoulder_height: If True, also require wrist/elbow at or above shoulder Y
    
    Returns:
        Tuple of (is_raised, angle)
    """
    shoulder = _get_keypoint(keypoints, shoulder_idx, kp_confidence_threshold)
    elbow = _get_keypoint(keypoints, elbow_idx, kp_confidence_threshold)
    wrist = _get_keypoint(keypoints, wrist_idx, kp_confidence_threshold)
    
    if shoulder is None or wrist is None:
        return False, 0.0
    
    # Vector from shoulder to wrist (Y inverted: top is 0)
    dx = wrist[0] - shoulder[0]
    dy = shoulder[1] - wrist[1]
    
    if abs(dx) < 1.0:  # Avoid division by zero
        return False, 0.0
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Arm is raised if angle is above threshold (pointing upward)
    angle_raised = angle_deg > arm_angle_threshold
    if not angle_raised:
        return False, angle_deg

    # Option A: also require wrist or elbow at or above shoulder height
    if require_shoulder_height and not _wrist_or_elbow_at_shoulder_height(shoulder, elbow, wrist):
        return False, angle_deg

    return True, angle_deg


def check_arm_raised(
    keypoints: List[List[float]],
    arm_angle_threshold: float,
    kp_confidence_threshold: float = 0.0,
    require_shoulder_height: bool = True,
) -> tuple[float, bool]:
    """
    Check if arm is raised (suspicious for weapon) — angle + optional shoulder-height.
    
    Args:
        keypoints: List of keypoints for one person [[x, y] or [x, y, conf], ...]
        arm_angle_threshold: Minimum angle (degrees) to consider arm raised
        kp_confidence_threshold: Min keypoint confidence (0 = use all)
        require_shoulder_height: If True, also require wrist/elbow at or above shoulder Y
    
    Returns:
        Tuple of (arm_angle, arm_raised)
    """
    if len(keypoints) < 11:  # Need at least wrist keypoints (index 10)
        return 0.0, False
    
    # Check both arms, use the more suspicious one
    left_raised, left_angle = check_single_arm(
        keypoints, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
        arm_angle_threshold, kp_confidence_threshold,
        require_shoulder_height=require_shoulder_height,
    )
    right_raised, right_angle = check_single_arm(
        keypoints, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
        arm_angle_threshold, kp_confidence_threshold,
        require_shoulder_height=require_shoulder_height,
    )
    
    if left_raised or right_raised:
        angle = left_angle if left_raised else right_angle
        return angle, True
    
    return 0.0, False


def analyze_arm_posture(
    pose_buffer: List[PoseFrame],
    temporal_consistency_frames: int,
    arm_angle_threshold: float,
    kp_confidence_threshold: float = 0.0,
    require_shoulder_height: bool = True,
) -> List[ArmPostureAnalysis]:
    """
    Analyze arm posture from buffered frames (angle + optional shoulder-height).
    
    Checks for raised arms (suspicious for weapon holding) with temporal consistency.
    
    Args:
        pose_buffer: Buffered pose frames
        temporal_consistency_frames: Minimum frames required for analysis
        arm_angle_threshold: Minimum angle (degrees) to consider arm raised
        kp_confidence_threshold: Min keypoint confidence (0 = use all)
        require_shoulder_height: If True, also require wrist/elbow at or above shoulder Y
    
    Returns:
        List of suspicious arm posture analyses
    """
    if len(pose_buffer) < temporal_consistency_frames:
        return []
    
    analyses = []
    latest_frame = pose_buffer[-1]
    
    # For each person in latest frame
    for person_idx, keypoints in enumerate(latest_frame.keypoints):
        if person_idx >= len(latest_frame.person_boxes):
            continue
        
        box = latest_frame.person_boxes[person_idx]
        
        # Check arm posture across buffered frames (temporal consistency)
        suspicious_count = 0
        arm_angles = []
        
        for frame in pose_buffer[-temporal_consistency_frames:]:
            if person_idx >= len(frame.keypoints):
                continue
            
            person_kp = frame.keypoints[person_idx]
            arm_angle, arm_raised = check_arm_raised(
                person_kp, arm_angle_threshold, kp_confidence_threshold,
                require_shoulder_height=require_shoulder_height,
            )
            
            if arm_raised:
                suspicious_count += 1
                arm_angles.append(arm_angle)
        
        # Require temporal consistency (suspicious in majority of frames)
        if suspicious_count >= (temporal_consistency_frames * 0.7):
            avg_angle = sum(arm_angles) / len(arm_angles) if arm_angles else 0.0
            confidence = suspicious_count / temporal_consistency_frames
            
            analysis = ArmPostureAnalysis(
                person_index=person_idx,
                box=box,
                arm_raised=True,
                arm_angle=avg_angle,
                confidence=confidence,
                timestamp=latest_frame.timestamp,
                frame_index=latest_frame.frame_index
            )
            analyses.append(analysis)
            
            # Print when suspicious arm posture is detected
            print(f"[WeaponDetectionScenario] ✅ Suspicious arm posture detected on frame {latest_frame.frame_index}: person {person_idx} (arm_angle: {avg_angle:.1f}°, confidence: {confidence:.2f})")
    
    return analyses
