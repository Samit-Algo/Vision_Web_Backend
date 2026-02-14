"""
Processing helpers — shared utilities for the vision pipeline.

Exposes the shared store registry so other modules can access the
multiprocessing shared memory without circular imports.
"""

# -----------------------------------------------------------------------------
# Shared store (used by main.py, workers, and API)
# -----------------------------------------------------------------------------
from .shared_store import get_shared_store, set_shared_store

__all__ = [
    "set_shared_store",
    "get_shared_store",
]
