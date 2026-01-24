"""
Frame Processor
---------------

Drawing utilities for bounding boxes and pose keypoints on frames.
Used by the pipeline runner and executor for visualization and event frames.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore[assignment]

import numpy as np


# COCO pose keypoint skeleton (YOLO pose uses same indices)
# Each tuple: (start_idx, end_idx) for drawing lines between keypoints
POSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # face
    (5, 6),                             # shoulders
    (5, 7), (7, 9),                     # left arm
    (6, 8), (8, 10),                    # right arm
    (5, 11), (6, 12), (11, 12),         # torso
    (11, 13), (13, 15),                 # left leg
    (12, 14), (14, 16),                 # right leg
]


def _ensure_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for frame drawing. Install opencv-python.")


def draw_bounding_boxes(
    frame: np.ndarray,
    detections: Dict[str, Any],
    rules: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draw bounding boxes, class labels, and scores on a frame.

    Args:
        frame: BGR image (numpy array, HxWx3)
        detections: Dict with "boxes", "classes", "scores" (lists)
        rules: List of rule config dicts (reserved for future use)

    Returns:
        New BGR image with boxes drawn (frame is copied; original unchanged).
    """
    _ensure_cv2()
    out = frame.copy()

    boxes = detections.get("boxes") or []
    classes = detections.get("classes") or []
    scores = detections.get("scores") or []

    color = (0, 255, 0)  # BGR green
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    for i in range(min(len(boxes), len(classes), len(scores))):
        box = boxes[i]
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cls_name = str(classes[i]) if i < len(classes) else ""
        score = float(scores[i]) if i < len(scores) else 0.0
        label = f"{cls_name} {score:.2f}" if cls_name else f"{score:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            out, label, (x1, y1 - 2),
            font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA
        )

    return out


def draw_pose_keypoints(
    frame: np.ndarray,
    detections: Dict[str, Any],
    rules: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draw pose keypoints and skeleton on a frame.

    Expects detections["keypoints"] as list of persons; each person is
    a list of [x, y] or [x, y, confidence] per keypoint (COCO 17 format).

    Args:
        frame: BGR image (numpy array, HxWx3)
        detections: Dict with "keypoints" (list of list of [x,y] or [x,y,c])
        rules: List of rule config dicts (reserved for future use)

    Returns:
        New BGR image with keypoints and skeleton drawn.
    """
    _ensure_cv2()
    out = frame.copy()

    keypoints_list = detections.get("keypoints") or []
    if not keypoints_list:
        return out

    point_color = (0, 255, 255)  # BGR yellow
    skeleton_color = (0, 255, 0)  # BGR green
    point_radius = 3
    skeleton_thickness = 2
    min_conf = 0.25

    for person_kps in keypoints_list:
        if not person_kps:
            continue
        kps = []
        for pt in person_kps:
            if pt is None or len(pt) < 2:
                kps.append(None)
                continue
            x, y = float(pt[0]), float(pt[1])
            conf = float(pt[2]) if len(pt) >= 3 else 1.0
            if conf < min_conf:
                kps.append(None)
            else:
                kps.append((int(round(x)), int(round(y))))

        for i, pt in enumerate(kps):
            if pt is not None:
                cv2.circle(out, pt, point_radius, point_color, -1, cv2.LINE_AA)

        for (i, j) in POSE_SKELETON:
            if i >= len(kps) or j >= len(kps):
                continue
            a, b = kps[i], kps[j]
            if a is not None and b is not None:
                cv2.line(out, a, b, skeleton_color, skeleton_thickness, cv2.LINE_AA)

    return out
