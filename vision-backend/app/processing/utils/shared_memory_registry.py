"""
Shared Memory Registry
=====================

Registry for cross-module access to shared memory store.

FastAPI `main.py` creates the multiprocessing.Manager().dict() shared_store used by:
- CameraPublisher (writes shared_store[camera_id])
- Task workers (write shared_store[task_id] for annotated frames)

Other parts of the backend (e.g. API/WebSocket endpoints) need read access.
We keep this tiny registry to avoid circular imports.
"""

from __future__ import annotations

from typing import Any, Optional

_shared_store: Optional[Any] = None


def set_shared_store(store: Any) -> None:
    """Set the shared memory store."""
    global _shared_store
    _shared_store = store


def get_shared_store() -> Optional[Any]:
    """Get the shared memory store."""
    return _shared_store
