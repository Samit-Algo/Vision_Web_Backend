"""
Video File Source
-----------------

Reads frames from a static video file (e.g. uploaded MP4).
Used when agent source_type is video_file / task has video_path.
No camera_id or shared_store involved.
"""

import os
import time
from typing import Optional

import numpy as np

from .data_models import FramePacket

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore[assignment]


class VideoFileSource:
    """
    Source that reads frames from a video file.
    Returns None from read_frame() at end of file (EOF).
    """

    def __init__(self, video_path: str):
        """
        Initialize video file source.
        Args:
            video_path: Absolute or relative path to video file.
        """
        self.video_path = os.path.abspath(video_path) if video_path else ""
        self._cap: Optional["cv2.VideoCapture"] = None
        self._frame_index = 0
        self._fps: Optional[float] = None
        self._start_monotonic = time.monotonic()

    def _open(self) -> bool:
        """Open video file. Returns True if opened successfully."""
        if not cv2:
            print("[VideoFileSource] ⚠️ OpenCV (cv2) not available. Cannot read video file.")
            return False
        if not self.video_path or not os.path.isfile(self.video_path):
            print(f"[VideoFileSource] ⚠️ File not found: {self.video_path}")
            return False
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            print(f"[VideoFileSource] ⚠️ Could not open video: {self.video_path}")
            return False
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_index = 0
        self._start_monotonic = time.monotonic()
        return True

    def read_frame(self) -> Optional[FramePacket]:
        """Read the next frame. Returns None at end of file."""
        if self._cap is None and not self._open():
            return None
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        if len(frame.shape) != 3:
            return None
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
            return None
        ts = time.monotonic() - self._start_monotonic
        packet = FramePacket(
            frame=frame,
            frame_index=self._frame_index,
            timestamp=ts,
            fps=self._fps,
            source_id=self.video_path,
        )
        self._frame_index += 1
        return packet

    def is_available(self) -> bool:
        """True while file is open and not yet at EOF (we don't know until we read)."""
        if self._cap is None:
            return self.video_path != "" and os.path.isfile(self.video_path)
        return self._cap.isOpened()

    def close(self) -> None:
        """Release video capture."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
