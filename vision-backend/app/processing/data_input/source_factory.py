"""
Source factory
--------------

Creates the correct frame source from task config:
- video_path or source_type "video_file" → VideoFileSource
- camera_id + shared_store → HubSource (RTSP via shared memory)
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict, Optional, Protocol

# -----------------------------------------------------------------------------
# Local
# -----------------------------------------------------------------------------
from .data_models import FramePacket
from .hub_source import HubSource
from .video_file_source import VideoFileSource

# -----------------------------------------------------------------------------
# Source protocol (what all sources must implement)
# -----------------------------------------------------------------------------


class Source(Protocol):
    """Interface for frame sources. Pipeline only needs read_frame() and is_available()."""

    def read_frame(self) -> Optional[FramePacket]:
        """Return the next frame, or None if not available / EOF."""
        ...

    def is_available(self) -> bool:
        """Return True if the source can provide frames."""
        ...


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def create_source(
    task: Dict[str, Any],
    shared_store: Optional[Dict[str, Any]] = None,
) -> Optional[Source]:
    """
    Create a frame source from task config.

    - If task has video_path or source_type "video_file" → VideoFileSource.
    - Else if task has camera_id and shared_store → HubSource (reads from CameraPublisher).
    - Otherwise returns None (missing video_path or camera_id or shared_store).
    """
    video_path = (task.get("video_path") or "").strip()
    source_type = (task.get("source_type") or "").strip().lower()
    is_video_file = bool(video_path) or source_type == "video_file"

    if is_video_file:
        if not video_path:
            print("[SourceFactory] ⚠️ source_type is video_file but video_path is missing.")
            return None
        return VideoFileSource(video_path=video_path)

    camera_id = (task.get("camera_id") or "").strip()
    if not camera_id:
        print("[SourceFactory] ⚠️ No camera_id and no video_path. Need one or the other.")
        return None
    if shared_store is None:
        print("[SourceFactory] ⚠️ No shared_store provided. RTSP source requires shared_store.")
        return None
    return HubSource(camera_id=camera_id, shared_store=shared_store)
