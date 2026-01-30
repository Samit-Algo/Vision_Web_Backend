"""
Loom Machine State Scenario
----------------------------

Scenario for detecting RUNNING/STOPPED state of loom machines
using motion-based analysis on per-loom ROIs.
"""

from app.processing.vision_tasks.tasks.loom_machine_state.scenario import LoomMachineStateScenario

__all__ = ["LoomMachineStateScenario"]
