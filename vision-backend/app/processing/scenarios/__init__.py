"""
Scenarios Module
----------------

Handles semantic event processing through stateful scenario processors.

Scenarios are stateful processors that:
- Receive per-frame context (detections, rule matches, frames)
- Buffer and reason over time
- Call additional models (VLM, face recognition) when needed
- Emit semantic events when ready (delayed decisions)

Each scenario defines its own internal DTOs and flow.
"""

from app.processing.scenarios.models import (
    BaseScenario,
    ScenarioEvent,
    ScenarioFrameContext,
)
from app.processing.scenarios.orchestrator import ScenarioEngine
from app.processing.scenarios.providers import (
    get_scenario_class,
    register_scenario,
    scenario_registry,
)

# Import all scenarios to trigger registration decorators
from app.processing.scenarios.weapon_detection import scenario as weapon_detection_scenario  # noqa: F401
from app.processing.scenarios.class_count import scenario as class_count_scenario  # noqa: F401
from app.processing.scenarios.loom_machine_state import scenario as loom_machine_state_scenario  # noqa: F401

try:
    from app.processing.scenarios.class_presence import scenario as class_presence_scenario  # noqa: F401
except ImportError:
    pass

__all__ = [
    "BaseScenario",
    "ScenarioFrameContext",
    "ScenarioEvent",
    "ScenarioEngine",
    "register_scenario",
    "get_scenario_class",
    "scenario_registry",
]
