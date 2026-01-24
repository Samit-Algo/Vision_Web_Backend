"""
Scenario Engine
---------------

Manages scenario instances and orchestrates per-frame processing.

Creates scenario instances from configuration and calls process() each frame.
"""

from typing import List, Dict, Any

from app.processing.scenarios.contracts import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent
)
from app.processing.scenarios.registry import get_scenario_class
from app.processing.pipeline.context import PipelineContext


class ScenarioEngine:
    """
    Manages scenario instances and orchestrates per-frame processing.
    
    Creates scenario instances from configuration and calls process() each frame.
    """
    
    def __init__(self, pipeline_context: PipelineContext):
        """
        Initialize scenario engine.
        
        Args:
            pipeline_context: Pipeline context (for scenario configs)
        """
        self.pipeline_context = pipeline_context
        self.scenarios: List[BaseScenario] = []
        self._initialize_scenarios()
    
    def _initialize_scenarios(self) -> None:
        """
        Create scenario instances from task configuration.
        
        Reads scenario configs from pipeline_context.task["scenarios"]
        Creates instances of appropriate scenario classes.
        
        Only scenarios explicitly configured in the task are initialized.
        Rules are handled separately by the rule engine.
        """
        scenario_configs = self.pipeline_context.task.get("scenarios", [])
        
        if not scenario_configs:
            # No scenarios configured - this is fine, rules will be used instead
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
    
    def process_frame(
        self,
        frame_context: ScenarioFrameContext
    ) -> List[ScenarioEvent]:
        """
        Process a frame through all enabled scenarios.
        
        Only processes scenarios that were configured in the task.
        Each scenario filters internally (e.g., class_presence only checks for configured class).
        
        Args:
            frame_context: Per-frame context from pipeline
        
        Returns:
            List of scenario events from all scenarios
        """
        if not self.scenarios:
            return []  # No scenarios configured - skip processing
        
        all_events = []
        for scenario in self.scenarios:
            try:
                # Each scenario's process() method filters internally
                # e.g., ClassPresenceScenario only emits if configured class is detected
                events = scenario.process(frame_context)
                if events:
                    all_events.extend(events)
            except Exception as exc:
                # Ignore scenario errors (log but don't crash pipeline)
                print(f"[ScenarioEngine] ⚠️  Error in scenario '{scenario.scenario_id}': {exc}")
        return all_events
    
    def reset_all(self) -> None:
        """
        Reset all scenario states (called on task restart).
        """
        for scenario in self.scenarios:
            try:
                scenario.reset()
            except Exception as exc:
                print(f"[ScenarioEngine] ⚠️  Error resetting scenario '{scenario.scenario_id}': {exc}")
