"""
Shared store registry (cross-module access)
===========================================

FastAPI main.py creates the multiprocessing.Manager().dict() shared_store used by:
- CameraPublisher (writes shared_store[camera_id])
- Agent workers (write shared_store[agent_id] for annotated frames)

Other parts of the backend (e.g. API/WebSocket endpoints) need read access.
We keep this tiny registry to avoid circular imports.
"""

from __future__ import annotations

from typing import Any, Optional


# ============================================================================
# REGISTRY
# ============================================================================

_shared_store: Optional[Any] = None


def set_shared_store(store: Any) -> None:
    """Set the global shared store instance."""
    global _shared_store
    _shared_store = store


def get_shared_store() -> Optional[Any]:
    """Get the global shared store instance."""
    return _shared_store
