"""
Motion Detection for Loom Machines (Industrial-Grade)
-----------------------------------------------------

Uses MOG2 background subtraction + morphological cleanup + motion ratio.
Stable for 24/7 RTSP: avoids light flicker, camera noise, and frame-diff flip-flop.
"""

from typing import Optional
import cv2
import numpy as np
from datetime import datetime

from app.processing.vision_tasks.tasks.loom_machine_state.types import MotionAnalysis


def extract_roi(frame: np.ndarray, roi: list[int]) -> np.ndarray:
    """
    Extract region of interest from frame.

    Args:
        frame: Full frame (H, W, 3) BGR
        roi: [x1, y1, x2, y2]

    Returns:
        Cropped ROI image
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))
    return frame[y1:y2, x1:x2].copy()


def create_mog2_subtractor(
    history: int = 500,
    var_threshold: int = 16,
    detect_shadows: bool = False,
) -> cv2.BackgroundSubtractorMOG2:
    """
    Create MOG2 background subtractor for industrial loom monitoring.

    Long history = stable background; shadows OFF = avoid light flicker.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )


def apply_morphological_clean(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Remove noise from foreground mask: Gaussian blur, erosion, dilation.
    Keeps only consistent motion clusters.
    """
    if ksize < 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    eroded = cv2.erode(blurred, kernel)
    dilated = cv2.dilate(eroded, kernel)
    return dilated


def compute_motion_ratio_mog2(
    roi_bgr: np.ndarray,
    mog2: cv2.BackgroundSubtractorMOG2,
    preprocess_blur_ksize: int = 5,
    morph_ksize: int = 3,
) -> float:
    """
    Compute motion_ratio = foreground_pixels / total_roi_pixels using MOG2.

    Pipeline: optional blur -> MOG2.apply -> morphological clean -> count fg pixels.
    """
    if roi_bgr.size == 0:
        return 0.0

    if preprocess_blur_ksize >= 3 and preprocess_blur_ksize % 2 == 1:
        roi_bgr = cv2.GaussianBlur(roi_bgr, (preprocess_blur_ksize, preprocess_blur_ksize), 0)

    fg_mask = mog2.apply(roi_bgr)
    fg_mask = apply_morphological_clean(fg_mask, morph_ksize)
    total = fg_mask.size
    if total == 0:
        return 0.0
    fg_count = np.count_nonzero(fg_mask)
    return float(fg_count) / float(total)


def detect_motion(
    frame: np.ndarray,
    roi: list[int],
    mog2: cv2.BackgroundSubtractorMOG2,
    motion_ratio_stopped: float,
    preprocess_blur_ksize: int,
    morph_ksize: int,
    loom_id: str,
    frame_index: int,
    timestamp: datetime,
) -> MotionAnalysis:
    """
    Detect motion in a loom ROI using MOG2 + morphological cleanup.

    Returns MotionAnalysis with motion_ratio; motion_detected = (motion_ratio > motion_ratio_stopped)
    for idle-sample buffer. State RUNNING/STOPPED is decided in state manager from rolling average.
    """
    roi_image = extract_roi(frame, roi)
    motion_ratio = compute_motion_ratio_mog2(
        roi_image,
        mog2,
        preprocess_blur_ksize=preprocess_blur_ksize,
        morph_ksize=morph_ksize,
    )
    motion_detected = motion_ratio > motion_ratio_stopped
    return MotionAnalysis(
        loom_id=loom_id,
        motion_detected=motion_detected,
        motion_ratio=motion_ratio,
        confidence=motion_ratio,
        timestamp=timestamp,
        frame_index=frame_index,
        motion_energy=motion_ratio,
        optical_flow_magnitude=None,
    )
