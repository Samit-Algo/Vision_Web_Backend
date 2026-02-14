"""
Vision tasks (scenarios)
------------------------

Stateful scenario processors: receive per-frame context, buffer, optionally call VLM/face,
emit events when ready. Each scenario type is registered via @register_scenario and
loaded by the pipeline from rule type (e.g. fall_detection, restricted_zone).
"""

# -----------------------------------------------------------------------------
# Data models and engine
# -----------------------------------------------------------------------------
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

# Import scenarios so @register_scenario decorators run
from app.processing.vision_tasks.tasks.weapon_detection import scenario as weapon_detection_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.sleep_detection import scenario as sleep_detection_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.class_count import scenario as class_count_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.loom_machine_state import scenario as loom_machine_state_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.box_count import scenario as box_count_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.fall_detection import scenario as fall_detection_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.fire_detection import scenario as fire_detection_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.restricted_zone import scenario as restricted_zone_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.wall_climb_detection import scenario as wall_climb_detection_scenario  # noqa: F401
from app.processing.vision_tasks.tasks.face_detection import scenario as face_detection_scenario  # noqa: F401

from app.processing.vision_tasks.tasks.class_presence import scenario as class_presence_scenario  # noqa: F401

__all__ = [
    "BaseScenario",
    "ScenarioFrameContext",
    "ScenarioEvent",
    "ScenarioEngine",
    "register_scenario",
    "get_scenario_class",
    "scenario_registry",
]
