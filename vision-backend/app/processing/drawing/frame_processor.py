"""
Frame Processor
---------------

Drawing utilities for bounding boxes and pose keypoints on frames.
Used by the pipeline runner and executor for visualization and event frames.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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


def _scale_coords(
    coords: List[List[float]], width: int, height: int
) -> List[Tuple[int, int]]:
    """Scale coordinates to pixel values. If max coord <= 1 assume normalized 0-1."""
    if not coords or width <= 0 or height <= 0:
        return []
    max_val = max(max(p[0], p[1]) for p in coords)
    if max_val <= 1.0:
        return [(int(p[0] * width), int(p[1] * height)) for p in coords]
    return [(int(p[0]), int(p[1])) for p in coords]


def draw_zone_polygon(
    frame: np.ndarray,
    zone_coordinates: Optional[List[List[float]]],
    width: int,
    height: int,
    color: Tuple[int, int, int] = (0, 0, 255),  # BGR red (outline)
    thickness: int = 2,
    fill_color: Optional[Tuple[int, int, int]] = (0, 0, 255),  # BGR red for fill
    fill_alpha: float = 0.25,  # 0–1: light red tint over the zone
) -> None:
    """
    Draw restricted zone polygon on frame (in-place): light red fill + red outline.
    zone_coordinates: list of [x,y] (normalized 0-1 or pixel).
    """
    _ensure_cv2()
    if not zone_coordinates or len(zone_coordinates) < 3:
        return
    pts = _scale_coords(zone_coordinates, width, height)
    if len(pts) < 3:
        return
    pts_arr = np.array(pts, dtype=np.int32)
    # Light red fill: blend a red overlay over the zone
    if fill_color is not None and fill_alpha > 0:
        overlay = np.zeros_like(frame)
        cv2.fillPoly(overlay, [pts_arr], fill_color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
    # Red outline so the zone boundary is clear
    cv2.polylines(frame, [pts_arr], isClosed=True, color=color, thickness=thickness)


def draw_zone_line(
    frame: np.ndarray,
    line_zone: Optional[Dict[str, Any]],
    width: int,
    height: int,
    color: Tuple[int, int, int] = (0, 255, 255),  # BGR yellow
    thickness: int = 2,
) -> None:
    """
    Draw zone line on frame (in-place).
    line_zone: {"type": "line", "coordinates": [[x1,y1], [x2,y2]]} (normalized or pixel).
    """
    _ensure_cv2()
    if not line_zone or line_zone.get("type") != "line":
        return
    coords = line_zone.get("coordinates")
    if not coords or len(coords) != 2:
        return
    pts = _scale_coords(coords, width, height)
    if len(pts) != 2:
        return
    cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color, thickness)


def draw_boxes_in_zone_red(
    frame: np.ndarray,
    boxes: List[List[float]],
    classes: List[Any],
    scores: List[float],
    in_zone_indices: List[int],
    color_in_zone: Tuple[int, int, int] = (0, 0, 255),  # BGR red
) -> None:
    """
    Overlay red bounding boxes only for indices in in_zone_indices (in-place).
    Call after draw_bounding_boxes so normal boxes are green and in-zone ones get red overlay.
    """
    _ensure_cv2()
    in_zone_set = set(in_zone_indices or [])
    if not in_zone_set:
        return
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    for i in in_zone_set:
        if i >= len(boxes) or len(boxes[i]) < 4:
            continue
        box = boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cls_name = str(classes[i]) if i < len(classes) else ""
        score = float(scores[i]) if i < len(scores) else 0.0
        label = f"{cls_name} {score:.2f}" if cls_name else f"{score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_in_zone, thickness)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color_in_zone, -1)
        cv2.putText(
            frame, label, (x1, y1 - 2),
            font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA
        )


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


def draw_box_count_annotations(
    frame: np.ndarray,
    track_info: List[Dict[str, Any]],
    line_zone: Optional[Dict[str, Any]],
    counts: Optional[Dict[str, int]],
    target_class: str,
    active_tracks_count: int = 0
) -> np.ndarray:
    """
    Draw box counting annotations with side-transition based colors.

    Color Logic (based on crossing direction):
    - GREEN (0, 255, 0):  Not crossed yet - object still on initial side
    - ORANGE (0, 165, 255): Crossed Side 2 → Side 1 (ENTRY/ADD direction)
    - YELLOW (0, 255, 255): Crossed Side 1 → Side 2 (EXIT/OUT direction)

    Draws:
    - Bounding boxes (color based on crossing direction)
    - Center points (filled circle, same color as box)
    - Track ID labels
    - "ADD!" or "OUT!" text when crossed
    - Count text at top

    Args:
        frame: BGR image (numpy array, HxWx3)
        track_info: List of track info dicts with keys: track_id, center, bbox, counted, direction
        line_zone: Line zone dict (not used for drawing, kept for compatibility)
        counts: Counts dict with entry_count, exit_count, boxes_counted, etc.
        target_class: Target class name (e.g., "box")
        active_tracks_count: Number of active tracks being tracked

    Returns:
        New BGR image with annotations drawn (frame is copied; original unchanged).
    """
    _ensure_cv2()
    out = frame.copy()

    if not track_info and not counts:
        return out

    # Color definitions (BGR format)
    COLOR_GREEN = (0, 255, 0)      # Not crossed yet
    COLOR_ORANGE = (0, 165, 255)   # Side 2 → Side 1 (ENTRY/ADD)
    COLOR_YELLOW = (0, 255, 255)   # Side 1 → Side 2 (EXIT/OUT)

    # Draw each tracked box with side-transition based colors
    for track in track_info or []:
        if not track.get("bbox") or len(track["bbox"]) < 4:
            continue

        x1, y1, x2, y2 = [int(coord) for coord in track["bbox"][:4]]
        track_id = track.get("track_id")
        counted = track.get("counted", False)
        direction = track.get("direction")  # 'entry' or 'exit' or None

        # Determine box color based on crossing direction
        if not counted:
            box_color = COLOR_GREEN
        elif direction == "entry":
            box_color = COLOR_ORANGE
        elif direction == "exit":
            box_color = COLOR_YELLOW
        else:
            box_color = COLOR_GREEN

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

        if track.get("center") and len(track["center"]) >= 2:
            center_x, center_y = [int(c) for c in track["center"][:2]]
            cv2.circle(out, (center_x, center_y), 5, box_color, -1)

        if track_id is not None:
            label = f"ID:{track_id}"
            cv2.putText(out, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        if counted and direction:
            if direction == "entry":
                direction_text = "ADD!"
                text_color = COLOR_ORANGE
            else:
                direction_text = "OUT!"
                text_color = COLOR_YELLOW
            cv2.putText(out, direction_text, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # Draw count at top
    if counts:
        entry_count = counts.get("entry_count", 0)
        exit_count = counts.get("exit_count", 0)
        add_text = f"ADD: {entry_count}"
        out_text = f"OUT: {exit_count}"
        tracking_text = f"TRACKING: {active_tracks_count}"
        cv2.putText(out, add_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ORANGE, 2)
        cv2.putText(out, out_text, (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)
        cv2.putText(out, tracking_text, (280, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)

    return out
