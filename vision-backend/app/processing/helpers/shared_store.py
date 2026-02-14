"""
Shared store registry
--------------------

Single place to get/set the multiprocessing shared memory store.

- main.py creates Manager().dict() at startup and calls set_shared_store().
- Pipeline workers write processed frames to shared_store[agent_id].
- API/WebSocket endpoints read from get_shared_store() to stream frames.

This registry avoids passing the store through every layer and prevents circular imports.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Optional

# -----------------------------------------------------------------------------
# Registry (module-level store reference)
# -----------------------------------------------------------------------------

_shared_store: Optional[Any] = None


def set_shared_store(store: Any) -> None:
    """Set the shared memory store (called once at app startup from main.py)."""
    global _shared_store
    _shared_store = store


def get_shared_store() -> Optional[Any]:
    """Return the shared memory store, or None if not yet initialized."""
    return _shared_store
