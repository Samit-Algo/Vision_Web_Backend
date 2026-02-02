"""
Sleep Analyzer
--------------

Step 2: Decide if a person looks "possibly sleeping" from pose only.

We use two simple rules (no eyes needed):
1. Lying down: torso (shoulder–hip line) is almost horizontal.
2. Standing sleep: head is tilted down (nose below shoulders) and body is still.

We need the same pose to appear in most of the last N frames (temporal consistency)
before we say "possibly sleeping" and send frames to the VLM.
"""

import math
from typing import List, Optional, Tuple

from app.processing.vision_tasks.tasks.sleep_detection.types import PoseFrame, SleepAnalysis


# YOLO/COCO pose keypoint indices (same as weapon_detection / fall_detection)
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


def _get_keypoint(
    person_keypoints: List[List[float]],
    idx: int,
    confidence_threshold: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """
    Get (x, y) for one keypoint if it exists and is confident enough.
    Keypoints can be [x, y] or [x, y, conf].
    """
    if idx >= len(person_keypoints):
        return None
    kp = person_keypoints[idx]
    if not kp or len(kp) < 2:
        return None
    if len(kp) >= 3 and confidence_threshold > 0 and kp[2] < confidence_threshold:
        return None
    return float(kp[0]), float(kp[1])


def _torso_angle_degrees(
    left_shoulder: Tuple[float, float],
    right_shoulder: Tuple[float, float],
    left_hip: Tuple[float, float],
    right_hip: Tuple[float, float],
) -> float:
    """
    Angle of torso (shoulder–hip line) from horizontal, in degrees.
    - 0 or 180 = lying (horizontal).
    - 90 = standing (vertical).
    """
    # Mid shoulder and mid hip
    sx = (left_shoulder[0] + right_shoulder[0]) / 2
    sy = (left_shoulder[1] + right_shoulder[1]) / 2
    hx = (left_hip[0] + right_hip[0]) / 2
    hy = (left_hip[1] + right_hip[1]) / 2
    # Vector from hip to shoulder (image Y increases downward)
    dx = sx - hx
    dy = sy - hy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    angle_rad = math.atan2(abs(dy), abs(dx))
    angle_deg = math.degrees(angle_rad)
    # 0 = horizontal (lying), 90 = vertical (standing)
    return angle_deg


def _is_lying(
    keypoints: List[List[float]],
    torso_angle_lying_deg: float,
    kp_conf: float,
) -> Tuple[bool, float]:
    """
    True if torso is nearly horizontal (person lying down).
    Returns (is_lying, torso_angle_deg).
    """
    ls = _get_keypoint(keypoints, LEFT_SHOULDER, kp_conf)
    rs = _get_keypoint(keypoints, RIGHT_SHOULDER, kp_conf)
    lh = _get_keypoint(keypoints, LEFT_HIP, kp_conf)
    rh = _get_keypoint(keypoints, RIGHT_HIP, kp_conf)
    if ls is None or rs is None or lh is None or rh is None:
        return False, 0.0
    angle = _torso_angle_degrees(ls, rs, lh, rh)
    # In our calc: 0 = horizontal (lying), 90 = vertical (standing).
    # So "lying" means torso angle from horizontal is small.
    is_lying = angle <= torso_angle_lying_deg
    return is_lying, angle


def _head_down_angle_degrees(
    keypoints: List[List[float]],
    kp_conf: float,
) -> Optional[float]:
    """
    Approximate "head down" angle: how far nose is below shoulder line (in degrees).
    Returns None if we can't compute; else angle in degrees (larger = more head down).
    Used for standing sleep: head tilted forward / chin down.
    """
    nose = _get_keypoint(keypoints, NOSE, kp_conf)
    ls = _get_keypoint(keypoints, LEFT_SHOULDER, kp_conf)
    rs = _get_keypoint(keypoints, RIGHT_SHOULDER, kp_conf)
    if nose is None or ls is None or rs is None:
        return None
    shoulder_y = (ls[1] + rs[1]) / 2
    # In image coords, Y increases downward, so nose below shoulder => nose_y > shoulder_y
    dy = nose[1] - shoulder_y
    dx = abs(nose[0] - (ls[0] + rs[0]) / 2)
    if dx < 1e-6:
        return 90.0 if dy > 0 else 0.0
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


def _nose_below_shoulder_px(
    keypoints: List[List[float]],
    kp_conf: float,
) -> Optional[float]:
    """
    How many pixels the nose is below the shoulder line (positive = head forward/down).
    Used as alternative/supplement to angle for standing sleep (chin-down posture).
    """
    nose = _get_keypoint(keypoints, NOSE, kp_conf)
    ls = _get_keypoint(keypoints, LEFT_SHOULDER, kp_conf)
    rs = _get_keypoint(keypoints, RIGHT_SHOULDER, kp_conf)
    if nose is None or ls is None or rs is None:
        return None
    shoulder_y = (ls[1] + rs[1]) / 2
    return float(nose[1] - shoulder_y)  # positive = nose below shoulder


def _average_motion_px(
    pose_buffer: List[PoseFrame],
    person_index: int,
    kp_indices: List[int],
    kp_conf: float,
) -> float:
    """
    Average movement (in pixels) of keypoints for this person across the last frames.
    Low motion = person is still (possible standing sleep).
    """
    if len(pose_buffer) < 2 or person_index >= len(pose_buffer[-1].keypoints):
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(pose_buffer) - 1):
        curr = pose_buffer[i]
        next_f = pose_buffer[i + 1]
        if person_index >= len(curr.keypoints) or person_index >= len(next_f.keypoints):
            continue
        kp_curr = curr.keypoints[person_index]
        kp_next = next_f.keypoints[person_index]
        for idx in kp_indices:
            p1 = _get_keypoint(kp_curr, idx, kp_conf)
            p2 = _get_keypoint(kp_next, idx, kp_conf)
            if p1 is not None and p2 is not None:
                total += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                count += 1
    if count == 0:
        return 0.0
    return total / count


def analyze_sleep_posture(
    pose_buffer: List[PoseFrame],
    temporal_consistency_frames: int,
    torso_angle_lying_deg: float,
    head_down_angle_deg: float,
    motion_threshold_px: float,
    kp_confidence_threshold: float,
    min_nose_below_shoulder_px: float = 0.0,
    head_down_majority_ratio: float = 0.5,
) -> List[SleepAnalysis]:
    """
    For each person in the latest frame, check if they look "possibly sleeping":
    - Lying: torso horizontal for most of the last N frames.
    - Standing sleep: head forward/down (angle OR nose below shoulder px) and low motion.

    Head posture: we use (1) head-down angle (nose below shoulder line) and/or
    (2) min_nose_below_shoulder_px so that "chin down / head forward" posture
    (standing sleep) is detected even with moderate tilt.
    """
    if len(pose_buffer) < temporal_consistency_frames:
        return []

    analyses = []
    latest = pose_buffer[-1]
    n_check = temporal_consistency_frames
    frames_to_check = pose_buffer[-n_check:]
    print(f"[SleepDetection SleepAnalyzer] 🔍 Analyzing {len(latest.keypoints)} person(s) over last {n_check} frames (frame_index={latest.frame_index})")

    for person_idx in range(len(latest.keypoints)):
        if person_idx >= len(latest.person_boxes):
            continue
        box = latest.person_boxes[person_idx]

        lying_count = 0
        head_down_still_count = 0
        head_angles_seen = []
        nose_below_px_seen = []

        for pf in frames_to_check:
            if person_idx >= len(pf.keypoints):
                continue
            kp = pf.keypoints[person_idx]

            # Rule 1: Lying
            is_lying, torso_angle = _is_lying(kp, torso_angle_lying_deg, kp_confidence_threshold)
            if is_lying:
                lying_count += 1

            # Rule 2: Head down/forward (angle OR nose below shoulder in pixels)
            head_angle = _head_down_angle_degrees(kp, kp_confidence_threshold)
            nose_below_px = _nose_below_shoulder_px(kp, kp_confidence_threshold)
            if head_angle is not None:
                head_angles_seen.append(head_angle)
            if nose_below_px is not None:
                nose_below_px_seen.append(nose_below_px)
            head_down_this_frame = False
            if head_angle is not None and head_angle >= head_down_angle_deg:
                head_down_this_frame = True
            if not head_down_this_frame and nose_below_px is not None and min_nose_below_shoulder_px > 0 and nose_below_px >= min_nose_below_shoulder_px:
                head_down_this_frame = True
            if head_down_this_frame:
                head_down_still_count += 1

        motion = _average_motion_px(
            pose_buffer,
            person_idx,
            [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
            kp_confidence_threshold,
        )
        is_still = motion <= motion_threshold_px

        # Lying: require 70% of frames. Head-down still: require head_down_majority_ratio (e.g. 50% = 2 of 4) AND still
        lying_majority = (n_check * 0.7)
        head_down_majority = max(1, int(n_check * head_down_majority_ratio))
        possibly_sleeping = False
        reason = ""
        confidence = 0.0

        if lying_count >= lying_majority:
            possibly_sleeping = True
            reason = "lying_down"
            confidence = lying_count / n_check
        elif head_down_still_count >= head_down_majority and is_still:
            possibly_sleeping = True
            reason = "head_down_still"
            confidence = (head_down_still_count / n_check) * 0.9
        elif is_still and head_down_still_count >= 1 and head_down_still_count < head_down_majority:
            # Standing/sitting sleep: slight head-down + very still (VLM will confirm)
            possibly_sleeping = True
            reason = "still_person"
            confidence = 0.5

        # Debug: log actual values so we can tune thresholds
        avg_head_angle = (sum(head_angles_seen) / len(head_angles_seen)) if head_angles_seen else None
        avg_nose_below_px = (sum(nose_below_px_seen) / len(nose_below_px_seen)) if nose_below_px_seen else None
        ha_str = f"{avg_head_angle:.1f}°" if avg_head_angle is not None else "n/a"
        np_str = f"{avg_nose_below_px:.1f}" if avg_nose_below_px is not None else "n/a"
        print(f"[SleepDetection SleepAnalyzer] person={person_idx} head_angle_avg={ha_str} nose_below_px_avg={np_str} motion={motion:.1f}px is_still={is_still} head_down_count={head_down_still_count}/{n_check} lying_count={lying_count} -> possibly_sleeping={possibly_sleeping}")

        if not possibly_sleeping:
            continue

        print(f"[SleepDetection SleepAnalyzer] ⚠️ Possibly sleeping: person_index={person_idx} reason={reason} confidence={confidence:.2f} lying_count={lying_count} head_down_still={head_down_still_count} is_still={is_still}")
        analyses.append(
            SleepAnalysis(
                person_index=person_idx,
                box=box,
                possibly_sleeping=True,
                reason=reason,
                confidence=confidence,
                timestamp=latest.timestamp,
                frame_index=latest.frame_index,
            )
        )

    return analyses
