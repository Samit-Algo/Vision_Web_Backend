"""
Rule registry
-------------

Provides a simple registry + decorator for rule handlers.
Each rule type registers a callable that evaluates a rule instance.
"""
from typing import Any, Callable, Dict, Optional

RuleHandler = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Any], Optional[Dict[str, Any]]]

rules_registry: Dict[str, RuleHandler] = {}


def register_rule(rule_type: str) -> Callable[[RuleHandler], RuleHandler]:
    """
    Decorator to register a rule handler under a specific type string.
    """
    def decorator(handler_function: RuleHandler) -> RuleHandler:
        rules_registry[rule_type] = handler_function
        return handler_function
    return decorator


