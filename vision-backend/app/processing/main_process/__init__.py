"""
Main process (vision runner)
----------------------------

Entry point for the background process that starts camera publishers and task workers.
"""

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
from .run_vision_app import main

__all__ = ["main"]
