"""
Scenario Registry
-----------------

Provides a simple registry + decorator for scenario classes.
Each scenario type registers a class that implements BaseScenario.

Registered scenario types (imported in vision_tasks/__init__.py):
- class_count      : Count detections of a class (optionally in zone)
- class_presence   : Detect presence of a class (optional)
- box_count        : Count boxes with optional line crossing / entry-exit
- fall_detection   : Detect human falls via pose keypoints
- fire_detection   : Detect fire/smoke in feed
- restricted_zone  : Alert when object of class is inside polygon zone
- wall_climb_detection : Alert when person climbs or is fully above wall (orange/red)
- loom_machine_state : Loom run/idle state from motion
- weapon_detection  : Weapon detection with pose + optional VLM
- sleep_detection   : Person sleeping (lying or standing) with pose + VLM confirmation
"""

from typing import Dict, Type

from app.processing.vision_tasks.data_models import BaseScenario

scenario_registry: Dict[str, Type[BaseScenario]] = {}


def register_scenario(scenario_type: str):
    """
    Decorator to register a scenario class under a specific type string.
    
    Usage:
        @register_scenario("weapon_detection")
        class WeaponDetectionScenario(BaseScenario):
            ...
    
    Args:
        scenario_type: Scenario type identifier (e.g., "weapon_detection")
    
    Returns:
        Decorator function
    """
    def decorator(scenario_class: Type[BaseScenario]) -> Type[BaseScenario]:
        scenario_registry[scenario_type] = scenario_class
        return scenario_class
    return decorator


def get_scenario_class(scenario_type: str) -> Type[BaseScenario] | None:
    """
    Get scenario class by type.
    
    Args:
        scenario_type: Scenario type identifier
    
    Returns:
        Scenario class or None if not found
    """
    return scenario_registry.get(scenario_type)
