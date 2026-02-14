"""
Hub source (shared memory)
--------------------------

Reads frames from shared memory. CameraPublisher writes RTSP frames to shared_store[camera_id];
HubSource reads the latest frame for the pipeline. Used for RTSP camera tasks.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict, Optional

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
import numpy as np

# -----------------------------------------------------------------------------
# Local
# -----------------------------------------------------------------------------
from .data_models import FramePacket

# -----------------------------------------------------------------------------
# Frame reconstruction (bytes + shape + dtype → numpy array)
# -----------------------------------------------------------------------------


def reconstruct_frame_from_entry(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """Rebuild a BGR frame from a shared_store entry. Returns None if entry is invalid."""
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


# Public alias for API/controllers that need to reconstruct frames from shared_store
reconstruct_frame = reconstruct_frame_from_entry

# -----------------------------------------------------------------------------
# Hub source
# -----------------------------------------------------------------------------


class HubSource:
    """
    Source that reads the latest frame from shared memory for a given camera_id.

    Used when the task is tied to an RTSP camera: CameraPublisher fills shared_store[camera_id],
    and HubSource.read_frame() returns that as a FramePacket.
    """

    def __init__(self, camera_id: str, shared_store: Dict[str, Any]) -> None:
        self.camera_id = camera_id
        self.shared_store = shared_store
        self._last_frame_index: Optional[int] = None

    def read_frame(self) -> Optional[FramePacket]:
        """Read the latest frame from shared_store[camera_id]. Returns None if no frame or error."""
        entry = self.shared_store.get(self.camera_id, {}) if self.shared_store else {}
        if not entry:
            return None

        frame = reconstruct_frame_from_entry(entry)
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
            source_id=self.camera_id,
        )

    def is_available(self) -> bool:
        """True if shared_store has a valid frame entry for this camera (no error)."""
        entry = self.shared_store.get(self.camera_id, {}) if self.shared_store else {}
        return bool(entry and "bytes" in entry)
