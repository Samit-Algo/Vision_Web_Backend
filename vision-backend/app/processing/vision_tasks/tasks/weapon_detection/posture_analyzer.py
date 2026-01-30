"""
Arm Posture Analyzer
--------------------

Analyzes arm posture from pose keypoints to detect suspicious patterns.
"""

from typing import List
import math

from app.processing.vision_tasks.tasks.weapon_detection.types import (
    PoseFrame,
    ArmPostureAnalysis
)


# YOLO pose keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10


def check_single_arm(
    keypoints: List[List[float]],
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
    arm_angle_threshold: float
) -> tuple[bool, float]:
    """
    Check if a single arm is raised.
    
    Args:
        keypoints: List of keypoints
        shoulder_idx: Shoulder keypoint index
        elbow_idx: Elbow keypoint index
        wrist_idx: Wrist keypoint index
        arm_angle_threshold: Minimum angle (degrees) to consider arm raised
    
    Returns:
        Tuple of (is_raised, angle)
    """
    if (shoulder_idx >= len(keypoints) or
        elbow_idx >= len(keypoints) or
        wrist_idx >= len(keypoints)):
        return False, 0.0
    
    shoulder = keypoints[shoulder_idx]
    elbow = keypoints[elbow_idx]
    wrist = keypoints[wrist_idx]
    
    if len(shoulder) < 2 or len(elbow) < 2 or len(wrist) < 2:
        return False, 0.0
    
    # Calculate angle of arm from horizontal
    # Vector from shoulder to wrist
    dx = wrist[0] - shoulder[0]
    dy = shoulder[1] - wrist[1]  # Y is inverted (top is 0)
    
    if abs(dx) < 1.0:  # Avoid division by zero
        return False, 0.0
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Arm is raised if angle is above threshold (pointing upward)
    is_raised = angle_deg > arm_angle_threshold
    
    return is_raised, angle_deg


def check_arm_raised(keypoints: List[List[float]], arm_angle_threshold: float) -> tuple[float, bool]:
    """
    Check if arm is raised (suspicious for weapon).
    
    Args:
        keypoints: List of keypoints for one person [[x, y], ...]
        arm_angle_threshold: Minimum angle (degrees) to consider arm raised
    
    Returns:
        Tuple of (arm_angle, arm_raised)
    """
    if len(keypoints) < 11:  # Need at least wrist keypoints
        return 0.0, False
    
    # Check both arms, use the more suspicious one
    left_raised, left_angle = check_single_arm(
        keypoints, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, arm_angle_threshold
    )
    right_raised, right_angle = check_single_arm(
        keypoints, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, arm_angle_threshold
    )
    
    if left_raised or right_raised:
        angle = left_angle if left_raised else right_angle
        return angle, True
    
    return 0.0, False


def analyze_arm_posture(
    pose_buffer: List[PoseFrame],
    temporal_consistency_frames: int,
    arm_angle_threshold: float
) -> List[ArmPostureAnalysis]:
    """
    Analyze arm posture from buffered frames.
    
    Checks for raised arms (suspicious for weapon holding).
    Requires temporal consistency across multiple frames.
    
    Args:
        pose_buffer: Buffered pose frames
        temporal_consistency_frames: Minimum frames required for analysis
        arm_angle_threshold: Minimum angle (degrees) to consider arm raised
    
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
        
        # Check arm posture across buffered frames
        suspicious_count = 0
        arm_angles = []
        
        for frame in pose_buffer[-temporal_consistency_frames:]:
            if person_idx >= len(frame.keypoints):
                continue
            
            person_kp = frame.keypoints[person_idx]
            arm_angle, arm_raised = check_arm_raised(person_kp, arm_angle_threshold)
            
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
