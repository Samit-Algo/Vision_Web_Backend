"""
Video I/O helpers
-----------------

Functions to open camera or file sources using OpenCV.
"""
from typing import Tuple, Optional

try:
    import cv2  # type: ignore
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore[assignment]


def open_video_capture(task: dict) -> Tuple[Optional["any"], Optional[str]]:
    """
    Open a video source for the given task.
    Preference order: source_uri (stream) -> file_path (file).
    Returns (cv2.VideoCapture or None, source_string or None).
    """
    if cv2 is None:
        print("⚠️ OpenCV not available. Skipping video capture.")
        return None, None
    src = task.get("source_uri") or task.get("file_path")
    if not src:
        print("⚠️ No video source configured on task. Skipping capture.")
        return None, None
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {src}")
        return None, src
    return cap, src


