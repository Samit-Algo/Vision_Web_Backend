"""
Source Factory
--------------

Factory to create frame source based on task config:
- If source is video file (video_path / source_type video_file) -> VideoFileSource
- Else if RTSP (camera_id) -> HubSource via shared memory
"""

from typing import Any, Dict, Optional, Protocol

from .data_models import FramePacket
from .hub_source import HubSource
from .video_file_source import VideoFileSource


# ============================================================================
# SOURCE PROTOCOL
# ============================================================================

class Source(Protocol):
    """
    Protocol defining the interface that all sources must implement.

    This allows the pipeline to work with any source type without knowing
    the specific implementation.
    """

    def read_frame(self) -> Optional[FramePacket]:
        """Read the next frame from the source."""
        ...

    def is_available(self) -> bool:
        """Check if source is available."""
        ...


# ============================================================================
# FACTORY
# ============================================================================

def create_source(
    task: Dict[str, Any],
    shared_store: Optional[Dict[str, Any]] = None
) -> Optional[Source]:
    """
    Create frame source from task config.
    - If task has video_path or source_type "video_file" -> VideoFileSource (no camera_id).
    - Else if task has camera_id and shared_store -> HubSource (RTSP).
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
