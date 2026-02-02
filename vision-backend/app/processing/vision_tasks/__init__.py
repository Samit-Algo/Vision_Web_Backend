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

from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioEvent,
    ScenarioFrameContext,
)
from app.processing.vision_tasks.setup_task import ScenarioEngine
from app.processing.vision_tasks.task_lookup import (
    get_scenario_class,
    register_scenario,
    scenario_registry,
)

# Import all scenarios to trigger registration decorators
from app.processing.vision_tasks.tasks.weapon_detection import scenario as weapon_detection_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.sleep_detection import scenario as sleep_detection_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.class_count import scenario as class_count_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.loom_machine_state import scenario as loom_machine_state_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.box_count import scenario as box_count_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.fall_detection import scenario as fall_detection_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.fire_detection import scenario as fire_detection_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.restricted_zone import scenario as restricted_zone_scenario  # noqa: F401

try:
    from app.processing.vision_tasks.tasks.class_presence import scenario as class_presence_scenario  # noqa: F401
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
