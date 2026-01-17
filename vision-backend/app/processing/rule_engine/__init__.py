"""
Rule Engine package
-------------------

Purpose:
- Define lightweight, readable rules per Agent.
- Evaluate rules against per-frame detections in the worker.
- Maintain minimal in-memory state (e.g., duration tracking).

Auto-registration:
- Import rule type modules here so they register themselves in the registry.
"""

# Ensure rule types register on package import
from app.processing.rule_engine.rule_types import class_presence  # noqa: F401
from app.processing.rule_engine.rule_types import count_at_least  # noqa: F401
from app.processing.rule_engine.rule_types import class_count  # noqa: F401
from app.processing.rule_engine.rule_types import accident_presence  # noqa: F401


