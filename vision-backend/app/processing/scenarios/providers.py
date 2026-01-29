"""
Scenario Providers Registry
---------------------------

Registers and resolves scenario classes by scenario type.
"""

from typing import Dict, Type

from app.processing.scenarios.models import BaseScenario

scenario_registry: Dict[str, Type[BaseScenario]] = {}


def register_scenario(scenario_type: str):
    """Decorator to register a scenario class under a type string."""

    def decorator(scenario_class: Type[BaseScenario]) -> Type[BaseScenario]:
        scenario_registry[scenario_type] = scenario_class
        return scenario_class

    return decorator


def get_scenario_class(scenario_type: str) -> Type[BaseScenario] | None:
    """Get scenario class by type."""
    return scenario_registry.get(scenario_type)

"""
Scenario Registry
-----------------

Provides a simple registry + decorator for scenario classes.
Each scenario type registers a class that implements BaseScenario.
"""

from typing import Dict, Type

from app.processing.scenarios.contracts import BaseScenario


# ============================================================================
# REGISTRY
# ============================================================================

scenario_registry: Dict[str, Type[BaseScenario]] = {}


def register_scenario(scenario_type: str):
    """
    Decorator to register a scenario class under a specific type string.

    Usage:
        @register_scenario("weapon_detection")
        class WeaponDetectionScenario(BaseScenario):
            ...
    """

    def decorator(scenario_class: Type[BaseScenario]) -> Type[BaseScenario]:
        scenario_registry[scenario_type] = scenario_class
        return scenario_class
    return decorator


def get_scenario_class(scenario_type: str) -> Type[BaseScenario] | None:
    """Get scenario class by type."""
    return scenario_registry.get(scenario_type)
