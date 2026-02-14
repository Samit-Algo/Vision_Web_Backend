"""
Video file source
-----------------

Reads frames from a single video file (e.g. uploaded MP4).
Used when task has video_path or source_type "video_file". read_frame() returns None at EOF.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import os
import time
from typing import Optional

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
import numpy as np

# -----------------------------------------------------------------------------
# Local
# -----------------------------------------------------------------------------
from .data_models import FramePacket

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Video file source
# -----------------------------------------------------------------------------


class VideoFileSource:
    """
    Source that reads frames from a video file.

    Opens the file on first read_frame(). Returns None at end of file (EOF).
    """

    def __init__(self, video_path: str) -> None:
        self.video_path = os.path.abspath(video_path) if video_path else ""
        self._cap: Optional["cv2.VideoCapture"] = None
        self._frame_index = 0
        self._fps: Optional[float] = None
        self._start_monotonic = time.monotonic()

    def open_capture(self) -> bool:
        """Open the video file. Returns True if opened successfully."""
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
        """Read the next frame. Returns None at EOF or on error."""
        if self._cap is None and not self.open_capture():
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
        """True while the file is open; we only know EOF after a failed read."""
        if self._cap is None:
            return self.video_path != "" and os.path.isfile(self.video_path)
        return self._cap.isOpened()

    def close(self) -> None:
        """Release the video capture."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
