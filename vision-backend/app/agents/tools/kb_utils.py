"""
Shared knowledge base utilities for agent tools.

This module provides centralized, data-driven logic for:
- Loading and querying the knowledge base
- Computing missing fields based on KB rules
- Applying rule defaults
- Validating run_mode against KB

All tools should import from here to ensure consistency.
"""
from __future__ import annotations

import json
from typing import Dict, List
from pathlib import Path

from ..session_state.agent_state import AgentState


# Load knowledge base once at module level
KB_PATH = Path(__file__).resolve().parent.parent.parent / "knowledge_base" / "vision_rule_knowledge_base.json"

with open(KB_PATH, "r", encoding="utf-8") as f:
    _KB_RULES: List[Dict] = json.load(f)["rules"]


def get_rule(rule_id: str) -> Dict:
    """
    Get a rule definition from the knowledge base by rule_id.
    
    Args:
        rule_id: The rule ID to look up (e.g., "class_presence", "gesture_detected")
        
    Returns:
        Rule dictionary from knowledge base
        
    Raises:
        ValueError: If rule_id is not found in knowledge base
    """
    for rule in _KB_RULES:
        if rule.get("rule_id") == rule_id:
            return rule
    raise ValueError(f"Unknown rule_id: {rule_id}")


def compute_missing_fields(agent: AgentState, rule: Dict) -> None:
    """
    Deterministic computation of missing fields.
    Fully data-driven from knowledge base - no hardcoding.
    
    This function:
    1. Starts with detection-specific required fields
    2. Adds execution mode required fields based on current run_mode
    3. Handles zone requirements (from rule OR execution mode)
    4. Handles time window requirements
    5. Validates and corrects invalid run_mode
    
    Args:
        agent: The agent state to compute missing fields for
        rule: The rule definition from knowledge base
    """
    # Build a deterministic ordered list of required fields.
    # IMPORTANT: Do NOT use a set here (set iteration order is non-deterministic),
    # otherwise the assistant will ask questions in a random order across turns.
    ordered_required: List[str] = []

    def add_unique(field_name: str) -> None:
        if field_name and field_name not in ordered_required:
            ordered_required.append(field_name)

    # 1) Detection-specific fields first (order as defined in KB)
    for f in rule.get("required_fields_from_user", []) or []:
        add_unique(f)

    # 2) Camera always required
    add_unique("camera_id")
    
    # Get zone support from KB (zone_support.required)
    zone_support = rule.get("zone_support", {})
    requires_zone = bool(zone_support.get("required", False))
    # NOTE: requires_zone is computed, NOT stored in agent.fields
    # It's derived state that should be recomputed from KB when needed
    
    # Get current run_mode (defaults to "continuous" if not set)
    run_mode = agent.fields.get("run_mode", "continuous")
    
    # Get execution modes from KB
    execution_modes = rule.get("execution_modes", {})
    
    # CRITICAL: Validate run_mode against KB - auto-correct if invalid
    if run_mode not in execution_modes:
        # Invalid run_mode - fall back to default from KB
        defaults = rule.get("defaults", {})
        run_mode = defaults.get("run_mode", "continuous")
        agent.fields["run_mode"] = run_mode
    
    # Get execution mode configuration from KB
    execution_config = execution_modes.get(run_mode, {})
    
    # Add execution mode required fields (data-driven, not hardcoded)
    execution_required = execution_config.get("required_fields", []) or []
    
    # Handle zone requirement (from rule OR execution mode)
    execution_zone_required = execution_config.get("zone_required", False)
    if execution_zone_required or requires_zone:
        add_unique("zone")

    # 3) Execution mode fields (e.g., patrol interval) after camera/zone
    for f in execution_required:
        add_unique(f)
    
    # 4) Time window fields last (keeps UX natural: what/where first, then when)
    time_window = rule.get("time_window", {})
    if time_window.get("required", False):
        add_unique("start_time")
        add_unique("end_time")
    # If time_window.supported is true but required is false, times are optional
    # LLM will only ask if user implies scheduling
    
    # Compute missing fields in deterministic order
    agent.missing_fields = [f for f in ordered_required if agent.fields.get(f) is None]
    
    print(f"[compute_missing_fields] Rule: {rule.get('rule_id')}, run_mode: {run_mode}")
    print(f"[compute_missing_fields] Ordered required fields: {ordered_required}")
    print(f"[compute_missing_fields] Agent fields: {list(agent.fields.keys())}")
    print(f"[compute_missing_fields] Missing fields: {agent.missing_fields}")


def apply_rule_defaults(agent: AgentState, rule: Dict) -> None:
    """
    Apply deterministic defaults from the rule definition.
    Dynamically applies ALL defaults from the knowledge base - no hardcoding.
    
    Args:
        agent: The agent state to apply defaults to
        rule: The rule definition from knowledge base
    """
    # Apply model (top-level in rule, not in defaults)
    if agent.fields.get("model") is None:
        agent.fields["model"] = rule.get("model")

    # Dynamically apply ALL defaults from the knowledge base
    defaults = rule.get("defaults", {})
    for field_name, default_value in defaults.items():
        # Only apply default if field is not already set
        if agent.fields.get(field_name) is None:
            agent.fields[field_name] = default_value


def compute_requires_zone(rule: Dict, run_mode: str) -> bool:
    """
    Compute whether zone is required based on KB structure.
    This is derived state - should NOT be stored in agent.fields.
    
    Args:
        rule: The rule definition from knowledge base
        run_mode: Current run_mode value
        
    Returns:
        True if zone is required, False otherwise
    """
    # Get zone support from KB (zone_support.required)
    zone_support = rule.get("zone_support", {})
    requires_zone = bool(zone_support.get("required", False))
    
    # Check execution mode zone requirement
    execution_modes = rule.get("execution_modes", {})
    execution_config = execution_modes.get(run_mode, {})
    execution_zone_required = execution_config.get("zone_required", False)
    
    return execution_zone_required or requires_zone

