"""
Source Factory
--------------

Factory to create HubSource for RTSP cameras via shared memory.
Only supports RTSP live streams (no file sources).
"""

from typing import Any, Dict, Optional, Protocol

from .data_models import FramePacket
from .hub_source import HubSource


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
    Create HubSource for RTSP camera via shared memory.

    Only supports RTSP live streams. Requires camera_id and shared_store.
    """
    camera_id = (task.get("camera_id") or "").strip()

    if not camera_id:
        print("[SourceFactory] ⚠️ No camera_id configured. RTSP source requires camera_id.")
        return None

    if shared_store is None:
        print("[SourceFactory] ⚠️ No shared_store provided. RTSP source requires shared_store.")
        return None

    return HubSource(camera_id=camera_id, shared_store=shared_store)
