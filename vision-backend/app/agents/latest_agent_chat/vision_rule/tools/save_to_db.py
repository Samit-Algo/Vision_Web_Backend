"""
Tool: save_to_db(). Persist rule config to MongoDB. Allowed only when status is confirmation.
Same validation and payload shape as ADK save_to_db. Human-in-the-loop intercepts before execution.
"""

import json
from datetime import datetime, timezone
from typing import Any

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..state import compute_requires_zone, get_rule
from .agent_persistence import (
    build_and_save_agent,
    is_persistence_available,
)
from .response import tool_response
from .time_utils import parse_natural_time


def build_rules_for_payload(rule: dict[str, Any], config: dict[str, Any], rule_id: str) -> list[dict[str, Any]]:
    """
    Build the rules array in ADK shape: one rule with "type" = rule_id and fields from rules_config_fields.
    Label is auto-generated from "class" if not provided. Zone included for class_count/box_count.
    """
    rule_config: dict[str, Any] = {"type": rule_id}
    rules_config_fields = rule.get("rules_config_fields") or []
    for field_name in rules_config_fields:
        if field_name == "label" and field_name not in config:
            if config.get("class"):
                class_name = str(config["class"]).replace("_", " ").title()
                rule_config["label"] = f"{class_name} detection"
        elif field_name in config and config[field_name] is not None:
            rule_config[field_name] = config[field_name]
    if rule_id in ("class_count", "box_count") and config.get("zone") is not None:
        rule_config["zone"] = config["zone"]
    return [rule_config]


def get_default_agent_name(rule: dict[str, Any], config: dict[str, Any]) -> str:
    """Default agent name when user did not set one: e.g. 'Person Class Presence Agent' or 'Fire Detection Agent'."""
    rule_name = rule.get("rule_name") or rule.get("rule_id") or "Agent"
    class_value = config.get("class") or config.get("gesture") or ""
    if class_value:
        class_label = str(class_value).replace("_", " ").title()
        return f"{class_label} {rule_name} Agent"
    return f"{rule_name} Agent"


def build_payload_from_state(state: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """
    Build Agent payload from runtime.state (config, rule_id, user_id). Same logic as ADK.
    Returns (payload_dict, None) on success, or (None, error_message) on validation failure.
    """
    config = state.get("config") or {}
    rule_id = state.get("rule_id") or ""
    user_id = state.get("user_id")
    missing_fields = state.get("missing_fields") or []

    if state.get("status") != "confirmation":
        return None, "Status must be 'confirmation' to save. Complete all required fields first."
    if missing_fields:
        return None, f"Missing required fields: {missing_fields}."
    if not user_id:
        return None, "User ID is required to save. Authentication error."

    run_mode = config.get("run_mode") or "continuous"
    if run_mode not in ("continuous", "patrol"):
        return None, "Invalid run mode. Use continuous or patrol."

    rule = get_rule(rule_id)
    if not rule:
        return None, "Rule not found."
    requires_zone = compute_requires_zone(rule, run_mode)

    start_time_str = config.get("start_time")
    end_time_str = config.get("end_time")
    start_time: datetime | None = None
    end_time: datetime | None = None
    try:
        if start_time_str:
            if isinstance(start_time_str, datetime):
                start_time = start_time_str
            else:
                start_time = parse_natural_time(
                    str(start_time_str), reference_datetime=datetime.now(timezone.utc)
                )
        if end_time_str:
            if isinstance(end_time_str, datetime):
                end_time = end_time_str
            else:
                end_time = parse_natural_time(
                    str(end_time_str),
                    reference_datetime=start_time or datetime.now(timezone.utc),
                )
    except (ValueError, TypeError) as error:
        return None, str(error) if isinstance(error, ValueError) else "Invalid time format. Use valid date and time."

    source_type = (config.get("source_type") or "").strip().lower()
    video_path = (config.get("video_path") or "").strip()
    is_video_file = source_type == "video_file" and bool(video_path)
    if is_video_file:
        start_time = None
        end_time = None

    # Build rules list in ADK shape: full rule config (type, class, label, etc.) from KB + config
    rules = config.get("rules")
    if not rules and rule_id:
        rules = build_rules_for_payload(rule, config, rule_id)

    # Model is required by Agent; take from config or rule
    model = config.get("model") or rule.get("model") or ""

    # Default name when empty (same as ADK: e.g. "Person Class Presence Agent")
    name = (config.get("name") or "").strip()
    if not name:
        name = get_default_agent_name(rule, config)

    payload: dict[str, Any] = {
        "name": name,
        "camera_id": config.get("camera_id") or "",
        "model": model,
        "fps": config.get("fps"),
        "rules": rules or [],
        "run_mode": run_mode,
        "start_time": start_time,
        "end_time": end_time,
        "requires_zone": requires_zone,
        "status": "PENDING",
        "owner_user_id": user_id,
        "video_path": video_path if is_video_file else "",
        "source_type": "video_file" if is_video_file else "rtsp",
    }

    if run_mode == "patrol":
        if config.get("interval_minutes") is not None:
            payload["interval_minutes"] = config["interval_minutes"]
        if config.get("check_duration_seconds") is not None:
            payload["check_duration_seconds"] = config["check_duration_seconds"]

    if requires_zone and config.get("zone") is not None:
        payload["zone"] = config["zone"]

    return payload, None


@tool
def save_to_db(runtime: ToolRuntime) -> Command:
    """
    Save the rule configuration to the database. Call when status is 'confirmation' (all required fields collected).
    Do not ask the user to confirm in chat—human approval is handled by the system.
    """
    state = runtime.state
    if state.get("status") != "confirmation":
        missing_fields = state.get("missing_fields") or []
        content = json.dumps(
            tool_response(
                status=state.get("status", "idle"),
                missing_fields=missing_fields,
                error="Cannot save. Status must be 'confirmation'. Missing fields: " + str(missing_fields) + ".",
            )
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    if not is_persistence_available():
        content = json.dumps(
            tool_response(
                status="confirmation",
                error="Persistence not available. Database not configured.",
            )
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    payload, error_message = build_payload_from_state(state)
    if error_message:
        content = json.dumps(
            tool_response(
                status="confirmation",
                error=error_message,
            )
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    if not payload:
        content = json.dumps(
            tool_response(status="confirmation", error="Failed to build save payload.")
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    try:
        saved = build_and_save_agent(payload)
    except ValueError:
        content = json.dumps(
            tool_response(
                status="confirmation",
                error="Invalid configuration. Check required fields (e.g. name, model, rules, camera or video).",
            )
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )
    except Exception:
        content = json.dumps(
            tool_response(
                status="confirmation",
                error="Failed to save agent. Please try again.",
            )
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    saved_id = getattr(saved, "id", None)
    saved_name = getattr(saved, "name", "")
    content = json.dumps(
        tool_response(
            status="completed",
            message="Agent saved successfully. Tell the user their agent is created.",
            final_payload={"agent_id": saved_id, "agent_name": saved_name},
        )
    )
    return Command(
        update={
            "status": "completed",
            "messages": [
                ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
            ],
        }
    )
