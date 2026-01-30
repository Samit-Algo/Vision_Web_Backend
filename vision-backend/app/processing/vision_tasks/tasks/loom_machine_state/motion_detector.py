"""
Motion Detection for Loom Machines
----------------------------------

Implements motion detection algorithms:
1. Frame difference (fast motion check)
2. Optical flow (motion confirmation)
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from datetime import datetime

from app.processing.vision_tasks.tasks.loom_machine_state.types import MotionAnalysis


def extract_roi(frame: np.ndarray, roi: list[int]) -> np.ndarray:
    """
    Extract region of interest from frame.
    
    Args:
        frame: Full frame image (H, W, 3) in BGR format
        roi: Region of interest [x1, y1, x2, y2]
    
    Returns:
        Cropped ROI image
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    
    # Clamp coordinates to frame bounds
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))
    
    # Extract ROI
    roi_image = frame[y1:y2, x1:x2]
    
    return roi_image


def compute_frame_difference(
    current_roi: np.ndarray,
    previous_roi: Optional[np.ndarray]
) -> Tuple[float, bool]:
    """
    Compute frame difference motion energy.
    
    Args:
        current_roi: Current ROI frame
        previous_roi: Previous ROI frame (None if first frame)
    
    Returns:
        Tuple of (motion_energy, motion_detected)
        motion_energy: Normalized motion energy (0.0-1.0)
        motion_detected: True if motion detected
    """
    if previous_roi is None:
        return 0.0, False
    
    # Ensure same size
    if current_roi.shape != previous_roi.shape:
        # Resize to match
        h, w = previous_roi.shape[:2]
        current_roi = cv2.resize(current_roi, (w, h))
    
    # Convert to grayscale if needed
    if len(current_roi.shape) == 3:
        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
    else:
        current_gray = current_roi
    
    if len(previous_roi.shape) == 3:
        previous_gray = cv2.cvtColor(previous_roi, cv2.COLOR_BGR2GRAY)
    else:
        previous_gray = previous_roi
    
    # Compute absolute difference
    diff = cv2.absdiff(current_gray, previous_gray)
    
    # Calculate motion energy (normalized)
    total_pixels = diff.size
    if total_pixels == 0:
        return 0.0, False
    
    motion_energy = np.sum(diff) / (total_pixels * 255.0)  # Normalize to [0.0, 1.0]
    
    return motion_energy, motion_energy > 0.0


def compute_optical_flow(
    current_roi: np.ndarray,
    previous_roi: Optional[np.ndarray],
    motion_threshold: float = 0.15
) -> Optional[float]:
    """
    Compute optical flow magnitude (only if fast motion check passes).
    
    Args:
        current_roi: Current ROI frame
        previous_roi: Previous ROI frame (None if first frame)
        motion_threshold: Motion threshold (skip if motion_energy < threshold)
    
    Returns:
        Average optical flow magnitude, or None if skipped
    """
    if previous_roi is None:
        return None
    
    # Fast motion check first (to reduce CPU usage)
    motion_energy, motion_detected = compute_frame_difference(current_roi, previous_roi)
    if not motion_detected or motion_energy < motion_threshold:
        return None  # Skip optical flow computation
    
    # Ensure same size
    if current_roi.shape != previous_roi.shape:
        h, w = previous_roi.shape[:2]
        current_roi = cv2.resize(current_roi, (w, h))
    
    # Convert to grayscale
    if len(current_roi.shape) == 3:
        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
    else:
        current_gray = current_roi
    
    if len(previous_roi.shape) == 3:
        previous_gray = cv2.cvtColor(previous_roi, cv2.COLOR_BGR2GRAY)
    else:
        previous_gray = previous_roi
    
    # Compute optical flow using Farneback method
    # Parameters optimized for loom motion detection
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,  # Image pyramid scale
        levels=3,  # Number of pyramid levels
        winsize=15,  # Averaging window size
        iterations=3,  # Number of iterations
        poly_n=5,  # Size of pixel neighborhood
        poly_sigma=1.2,  # Gaussian sigma
        flags=0
    )
    
    # Calculate magnitude of flow vectors
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Average magnitude
    avg_magnitude = np.mean(magnitude)
    
    return float(avg_magnitude)


def detect_motion(
    frame: np.ndarray,
    roi: list[int],
    previous_roi: Optional[np.ndarray],
    motion_threshold: float,
    optical_flow_threshold: float,
    loom_id: str,
    frame_index: int,
    timestamp: datetime
) -> MotionAnalysis:
    """
    Detect motion in a loom ROI using frame difference and optical flow.
    
    Args:
        frame: Current frame (full frame)
        roi: Region of interest [x1, y1, x2, y2]
        previous_roi: Previous ROI frame (None if first frame)
        motion_threshold: Frame difference threshold
        optical_flow_threshold: Optical flow magnitude threshold
        loom_id: Loom identifier
        frame_index: Current frame index
        timestamp: Current timestamp
    
    Returns:
        MotionAnalysis result
    """
    # Extract ROI
    current_roi = extract_roi(frame, roi)
    
    # Step 1: Fast motion check (frame difference)
    motion_energy, motion_detected_fast = compute_frame_difference(current_roi, previous_roi)
    
    # Step 2: Optical flow confirmation (only if fast check passes)
    optical_flow_magnitude = None
    motion_detected = False
    confidence = 0.0
    
    # Use a lower threshold for fast check to trigger optical flow computation
    # This allows us to catch subtle motion that might not pass the main threshold
    fast_check_threshold = motion_threshold * 0.5  # 50% of main threshold for fast check
    
    if motion_detected_fast and motion_energy >= fast_check_threshold:
        # Compute optical flow (use lower threshold to catch more motion)
        optical_flow_magnitude = compute_optical_flow(
            current_roi,
            previous_roi,
            fast_check_threshold  # Use lower threshold for optical flow trigger
        )
        
        if optical_flow_magnitude is not None:
            # Motion detected if EITHER motion_energy OR optical flow passes threshold
            # This makes detection more lenient for subtle motion
            motion_detected = (motion_energy >= motion_threshold) or (optical_flow_magnitude >= optical_flow_threshold)
            
            # Calculate confidence based on both metrics
            motion_confidence = min(1.0, motion_energy / motion_threshold) if motion_threshold > 0 else 0.0
            flow_confidence = min(1.0, optical_flow_magnitude / optical_flow_threshold) if optical_flow_threshold > 0 else 0.0
            confidence = (motion_confidence + flow_confidence) / 2.0
        else:
            # Optical flow computation skipped (low motion), but frame diff passed
            motion_detected = True  # If frame diff passed, consider it motion
            confidence = min(1.0, motion_energy / motion_threshold) if motion_threshold > 0 else 0.0
    else:
        # Fast check failed
        motion_detected = False
        confidence = 0.0
    
    return MotionAnalysis(
        loom_id=loom_id,
        motion_detected=motion_detected,
        motion_energy=motion_energy,
        optical_flow_magnitude=optical_flow_magnitude,
        confidence=confidence,
        timestamp=timestamp,
        frame_index=frame_index
    )
