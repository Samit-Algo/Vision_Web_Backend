"""
Class Count Scenario
--------------------

Simple scenario that counts detections of a specified class.
Optionally filters by zone and generates statistics reports.
"""

from typing import List, Dict, Any

from app.processing.scenarios.contracts import (
    BaseScenario,
    ScenarioFrameContext,
    ScenarioEvent
)
from app.processing.scenarios.registry import register_scenario
from app.processing.scenarios.class_count.config import ClassCountConfig
from app.processing.scenarios.class_count.counter import (
    count_class_detections,
    generate_count_label
)
from app.processing.scenarios.class_count.reporter import generate_report


@register_scenario("class_count")
class ClassCountScenario(BaseScenario):
    """
    Simple scenario that counts class detections.
    
    Stateless per-frame, but maintains statistics in state.
    """
    
    def __init__(self, config: Dict[str, Any], pipeline_context):
        super().__init__(config, pipeline_context)
        
        # Load configuration
        self.config_obj = ClassCountConfig(config, pipeline_context.task)
    
    def process(self, frame_context: ScenarioFrameContext) -> List[ScenarioEvent]:
        """
        Process a single frame.
        
        Only processes if target class is configured.
        Returns empty list immediately if no target class configured.
        
        Always returns count event (including zero) if target class is configured.
        """
        # Early exit: No target class configured
        if not self.config_obj.target_class:
            return []
        
        # Count detections
        matched_count, matched_indices = count_class_detections(
            frame_context.detections,
            self.config_obj.target_class,
            self.config_obj.zone_coordinates
        )
        
        # Generate report
        report = generate_report(
            self._state,
            matched_count,
            frame_context.timestamp,
            self.config_obj.zone_applied
        )
        
        # Generate label
        label = generate_count_label(
            matched_count,
            self.config_obj.target_class,
            self.config_obj.custom_label
        )
        
        # Emit event
        event = ScenarioEvent(
            event_type="class_count",
            label=label,
            confidence=1.0,  # Counts are deterministic
            metadata={
                "count": matched_count,
                "target_class": self.config_obj.target_class,
                "zone_applied": self.config_obj.zone_applied,
                "report": report
            },
            detection_indices=matched_indices,
            timestamp=frame_context.timestamp,
            frame_index=frame_context.frame_index
        )
        
        return [event]
    
    def reset(self) -> None:
        """Reset scenario state."""
        self._state.clear()
