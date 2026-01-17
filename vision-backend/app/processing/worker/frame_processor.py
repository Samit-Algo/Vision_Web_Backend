"""
Frame Processor
===============

Utilities for processing frames with bounding boxes based on agent rules.
"""
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None


def draw_bounding_boxes(
    frame: np.ndarray,
    detections: Dict[str, Any],
    rules: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Draw bounding boxes on frame based on agent rules.
    Only draws boxes for classes that match the agent's rules.
    
    Args:
        frame: OpenCV frame (numpy array in BGR format)
        detections: Dict with 'boxes', 'classes', 'scores'
        rules: List of agent rules
    
    Returns:
        Frame with bounding boxes drawn (same format as input)
    """
    if cv2 is None:
        print("[frame_processor] ⚠️  OpenCV not available, cannot draw bounding boxes")
        return frame
    
    # Extract target classes from rules
    target_classes = set()
    for rule in rules:
        rule_type = rule.get("type", "").lower()
        rule_class = rule.get("class", "").lower()
        
        if rule_type in ["class_presence", "count_at_least", "class_count"]:
            if rule_class:
                target_classes.add(rule_class)
    
    # If no target classes found, don't draw anything
    if not target_classes:
        return frame
    
    boxes = detections.get("boxes", [])
    classes = detections.get("classes", [])
    scores = detections.get("scores", [])
    
    # Make a copy to avoid modifying original
    processed_frame = frame.copy()
    
    # Draw boxes only for target classes
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        if cls.lower() in target_classes:
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within frame bounds
            height, width = processed_frame.shape[:2]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Draw rectangle (green color, thickness 2)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with class and score
            label = f"{cls} {score:.2f}" if score else cls
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle for text
            label_y = max(y1, text_height + 10)
            cv2.rectangle(
                processed_frame,
                (x1, label_y - text_height - 10),
                (x1 + text_width, label_y),
                (0, 255, 0),
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                processed_frame,
                label,
                (x1, label_y - 5),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                thickness
            )
    
    return processed_frame



# ========================================================================
# YOLOv8 Pose Detection
# ========================================================================
 
def draw_pose_keypoints(
    frame: np.ndarray,
    detections: Dict[str, Any],
    rules: List[Dict[str, Any]]
) -> np.ndarray:
    if cv2 is None:
        return frame
    keypoints = detections.get("keypoints") or []
    if not keypoints:
        return frame
    target_person = False
    for rule in rules or []:
        rule_type = str(rule.get("type", "")).lower()
        if rule_type == "accident_presence":
            target_person = True
            break
        if rule_type in ["class_presence", "count_at_least", "class_count"]:
            rule_class = str(rule.get("class", "")).lower()
            if rule_class == "person":
                target_person = True
                break
            classes = [str(c).lower() for c in (rule.get("classes") or []) if isinstance(c, str)]
            if "person" in classes:
                target_person = True
                break
    if not target_person:
        return frame
 
    processed_frame = frame.copy()
    height, width = processed_frame.shape[:2]
 
    skeleton = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)
    ]
 
    for person in keypoints:
        pts: List[Optional[tuple[int, int]]] = []
        for kp in person:
            if kp is None or len(kp) < 2:
                pts.append(None)
                continue
            x = int(max(0, min(int(kp[0]), width - 1)))
            y = int(max(0, min(int(kp[1]), height - 1)))
            pts.append((x, y))
 
        for p in pts:
            if p is None:
                continue
            cv2.circle(processed_frame, p, 3, (0, 255, 255), -1)
 
        for a, b in skeleton:
            pa = pts[a] if a < len(pts) else None
            pb = pts[b] if b < len(pts) else None
            if pa is None or pb is None:
                continue
            cv2.line(processed_frame, pa, pb, (0, 255, 0), 2)
 
    return processed_frame
 
