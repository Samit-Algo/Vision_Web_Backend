"""
Hub Source
----------

Reads frames from shared memory (populated by CameraPublisher).
Used when agent is connected to an RTSP camera via the hub.
"""

from typing import Any, Dict, Optional

import numpy as np

from app.processing.sources.contracts import FramePacket


# ============================================================================
# FRAME RECONSTRUCTION
# ============================================================================

def _reconstruct_frame(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """Rebuild a numpy array from the shared_store entry. Returns None if invalid."""
    try:
        if not entry or "bytes" not in entry or "shape" not in entry or "dtype" not in entry:
            return None
        buffer_bytes = entry["bytes"]
        shape: tuple[int, int, int] = tuple(entry["shape"])  # type: ignore[assignment]
        dtype = np.dtype(entry["dtype"])
        flat_array = np.frombuffer(buffer_bytes, dtype=dtype)
        expected_size = int(shape[0]) * int(shape[1]) * int(shape[2])
        if flat_array.size != expected_size:
            return None
        return flat_array.reshape(shape)
    except Exception:
        return None


# ============================================================================
# HUB SOURCE
# ============================================================================

class HubSource:
    """
    Source that reads frames from shared memory (hub).

    Used when agent is processing frames from an RTSP camera that is being
    published to shared_store by CameraPublisher.
    """

    def __init__(self, camera_id: str, shared_store: Dict[str, Any]):
        """Initialize hub source."""
        self.camera_id = camera_id
        self.shared_store = shared_store
        self._last_frame_index: Optional[int] = None

    def read_frame(self) -> Optional[FramePacket]:
        """Read the latest frame from shared memory."""
        entry = self.shared_store.get(self.camera_id, {}) if self.shared_store else {}
        if not entry:
            return None

        frame = _reconstruct_frame(entry)
        if frame is None:
            return None

        frame_index = int(entry.get("frame_index", 0)) if isinstance(entry, dict) else 0
        timestamp = float(entry.get("ts_monotonic", 0.0)) if isinstance(entry, dict) else 0.0
        fps = entry.get("camera_fps") if isinstance(entry, dict) else None
        if fps is not None:
            try:
                fps = float(fps)
            except (ValueError, TypeError):
                fps = None

        return FramePacket(
            frame=frame,
            frame_index=frame_index,
            timestamp=timestamp,
            fps=fps,
            source_id=self.camera_id
        )

    def is_available(self) -> bool:
        """Check if source is available (has frames in shared_store)."""
        entry = self.shared_store.get(self.camera_id, {}) if self.shared_store else {}
        return bool(entry and "bytes" in entry)
