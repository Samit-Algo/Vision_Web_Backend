"""
Scenario engine
---------------

Creates scenario instances from task config and runs process() per frame.
Used when scenarios are configured under task["scenarios"] (optional alternative to rule-based).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from app.processing.pipeline.context import PipelineContext
from app.processing.vision_tasks.data_models import (
    BaseScenario,
    ScenarioEvent,
    ScenarioFrameContext,
)
from app.processing.vision_tasks.task_lookup import get_scenario_class

# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------


class ScenarioEngine:
    """
    Manages scenario instances and orchestrates per-frame processing.

    Creates scenario instances from configuration and calls process() each frame.
    """

    def __init__(self, pipeline_context: PipelineContext):
        """Initialize scenario engine."""
        self.pipeline_context = pipeline_context
        self.scenarios: List[BaseScenario] = []
        self.initialize_scenarios()

    def initialize_scenarios(self) -> None:
        """Create scenario instances from task configuration."""
        scenario_configs = self.pipeline_context.task.get("scenarios", [])

        if not scenario_configs:
            return

        initialized_count = 0
        for config in scenario_configs:
            if not config.get("enabled", True):
                continue

            scenario_type = config.get("type")
            if not scenario_type:
                continue

            scenario_class = get_scenario_class(scenario_type)
            if scenario_class:
                try:
                    instance = scenario_class(config, self.pipeline_context)
                    self.scenarios.append(instance)
                    initialized_count += 1
                except Exception as exc:
                    print(f"[ScenarioEngine] ⚠️  Failed to initialize scenario '{scenario_type}': {exc}")

        if initialized_count > 0:
            print(f"[ScenarioEngine] Initialized {initialized_count} scenario(s) for task '{self.pipeline_context.task_id}'")

    def process_frame(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """Process a frame through all enabled scenarios."""
        if not self.scenarios:
            return []

        all_events = []
        for scenario in self.scenarios:
            try:
                events = scenario.process(frame_context)
                if events:
                    all_events.extend(events)
            except Exception as exc:
                print(f"[ScenarioEngine] ⚠️  Error in scenario '{scenario.scenario_id}': {exc}")
        return all_events

    def reset_all(self) -> None:
        """Reset all scenario states (called on task restart)."""
        for scenario in self.scenarios:
            try:
                scenario.reset()
            except Exception as exc:
                print(f"[ScenarioEngine] ⚠️  Error resetting scenario '{scenario.scenario_id}': {exc}")
