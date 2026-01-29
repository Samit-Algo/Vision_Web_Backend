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

from app.processing.scenarios.contracts import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent
)
from app.processing.scenarios.registry import (
    register_scenario,
    get_scenario_class,
    scenario_registry
)
from app.processing.scenarios.engine import ScenarioEngine

# Import all scenarios to trigger registration decorators
# This ensures @register_scenario decorators execute when the module is imported
from app.processing.scenarios.weapon_detection import scenario as weapon_detection_scenario  # noqa: F401
from app.processing.scenarios.class_count import scenario as class_count_scenario  # noqa: F401
from app.processing.scenarios.box_count import scenario as box_count_scenario  # noqa: F401
from app.processing.scenarios.restricted_zone import scenario as restricted_zone_scenario  # noqa: F401
from app.processing.scenarios.fall_detection import scenario as fall_detection_scenario  # noqa: F401
from app.processing.scenarios.fire_detection import scenario as fire_detection_scenario  # noqa: F401

# Check for class_presence scenario if it exists
try:
    from app.processing.scenarios.class_presence import scenario as class_presence_scenario  # noqa: F401
except ImportError:
    pass  # class_presence might not have a scenario.py yet

__all__ = [
    "BaseScenario",
    "ScenarioFrameContext",
    "ScenarioEvent",
    "ScenarioEngine",
    "register_scenario",
    "get_scenario_class",
    "scenario_registry",
]
