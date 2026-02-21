"""
Tool: initialize_state(rule_id). Start configuring a rule. Allowed only when no rule is set.
When the next required field is camera_id, returns the list of available cameras so the LLM
can show them and ask the user to choose; the user's choice is then set via set_field_value.
"""

import json

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..state import (
    build_initial_config,
    compute_missing_fields,
    get_next_required_field,
    get_rule,
    list_rule_ids,
)
from .camera_provider import get_cameras
from .response import tool_response, get_current_time_format_hint


def get_session_id_from_runtime(runtime: ToolRuntime) -> str:
    """Return session_id from state or from configurable.thread_id."""
    state = runtime.state
    state_session_id = state.get("session_id")
    if state_session_id:
        return state_session_id
    config = getattr(runtime, "config", None) or {}
    return (config.get("configurable") or {}).get("thread_id") or "default"


@tool
def initialize_state(rule_id: str, runtime: ToolRuntime) -> Command:
    """
    Start configuring a vision rule. Call once when the user says which rule they want (e.g. fire detection, class presence).
    Use rule_id from the knowledge base. Do not call if a rule is already being configured.
    When the next required field is camera_id, the tool returns a list of available cameras; show that list to the user and ask them to choose one, then call set_field_value(field_name="camera_id", value=<user's choice>).
    """
    state = runtime.state
    if state.get("rule_id") is not None:
        content = json.dumps(
            tool_response(
                status=state.get("status", "idle"),
                error="Already initialized. Complete or reset the current rule first.",
            )
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    rule = get_rule(rule_id)
    if not rule:
        content = json.dumps(
            tool_response(
                status="idle",
                error=f"Unknown rule_id: {rule_id}. Valid rule_ids: {list_rule_ids()}.",
            )
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    config = build_initial_config(rule)
    missing_fields = compute_missing_fields(rule, config)
    next_required_field = get_next_required_field(missing_fields)
    status = "confirmation" if not missing_fields else "collecting"

    cameras: list[dict[str, str]] = []
    user_id = state.get("user_id")
    session_id = get_session_id_from_runtime(runtime)
    if next_required_field == "camera_id" and user_id:
        cameras = get_cameras(user_id=user_id, session_id=session_id)

    if next_required_field == "camera_id" and cameras:
        message = (
            f"Rule '{rule.get('rule_name', rule_id)}' loaded. The next required field is camera_id. "
            "Show the user the 'cameras' list below and ask them to choose one by name or ID. "
            "When they reply, call set_field_value(field_name='camera_id', value=<their choice>)."
        )
    elif next_required_field in ("start_time", "end_time"):
        time_hint = get_current_time_format_hint()
        message = (
            f"Rule '{rule.get('rule_name', rule_id)}' loaded. The next required field(s): {', '.join(missing_fields)}. "
            f"{time_hint}"
        )
    elif next_required_field:
        message = (
            f"Rule '{rule.get('rule_name', rule_id)}' loaded. Ask the user only for the field: {next_required_field}."
        )
    else:
        message = "All required fields have defaults. Call save_to_db(); human approval will be requested."

    content = json.dumps(
        tool_response(
            status=status,
            next_required_field=next_required_field,
            missing_fields=missing_fields,
            message=message,
            config_preview=config if status == "confirmation" else None,
            cameras=cameras if cameras else None,
        )
    )
    return Command(
        update={
            "rule_id": rule_id,
            "config": config,
            "missing_fields": missing_fields,
            "status": status,
            "messages": [
                ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
            ],
        }
    )
