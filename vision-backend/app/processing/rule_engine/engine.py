"""
Unified rule engine
-------------------

Engine is rule-agnostic. It delegates rule evaluation to handlers registered
in rules_registry via the register_rule decorator.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import sys

# Allow running this file directly by ensuring project root is on sys.path
if __package__ is None or __package__ == "":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

from app.processing.rule_engine.registry import rules_registry


def evaluate_rules(rules: List[Dict[str, Any]], detections: Dict[str, Any], task: Dict[str, Any], state: Dict[int, Dict[str, Any]], now: datetime) -> Optional[Dict[str, Any]]:
    """
    Evaluate a list of rules (first match wins).

    - rules: list of rule dicts (loaded once at worker start)
    - detections: {'classes': [...], 'scores': [...], 'boxes': [...], 'ts': datetime}
    - task: full task dict (available to handlers)
    - state: dict indexed by rule index, each value is a per-rule state dict
    - now: current timestamp

    Returns:
      dict {'label': str, 'rule_index': int} or None
    """
    # Debug: Log available rules in registry (first call only)
    if not hasattr(evaluate_rules, '_logged_registry'):
        print(f"[evaluate_rules] Available rules in registry: {list(rules_registry.keys())}")
        evaluate_rules._logged_registry = True
    
    for rule_index, rule in enumerate(rules or []):
        rule_type = (rule.get("type") or "").strip().lower()
        handler = rules_registry.get(rule_type)
        if handler is None:
            print(f"[evaluate_rules] ⚠️ Rule type '{rule_type}' not found in registry. Available: {list(rules_registry.keys())}")
            continue
        rule_state = state.setdefault(rule_index, {"last_matched_since": None})
        evaluation_result = handler(rule, detections, task, rule_state, now)
        if evaluation_result and isinstance(evaluation_result, dict) and evaluation_result.get("label"):
            # Ensure rule_index is set
            evaluation_result.setdefault("rule_index", rule_index)
            return evaluation_result
    return None


