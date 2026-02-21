"""
Tool: set_field_value(field_name, value). Set one required field. Allowed only when status is collecting and field in missing_fields.
For field_name 'camera_id', value (user's choice by name or ID) is resolved to a camera_id before storing.
For start_time/end_time, natural language (e.g. "now", "in 10 min", "tomorrow 3pm") is parsed and stored as ISO.
"""

import json
from datetime import datetime, timezone

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..state import compute_missing_fields, get_next_required_field, get_rule
from .camera_provider import get_cameras, resolve_camera as resolve_camera_provider
from .response import tool_response, get_current_time_format_hint
from .time_utils import parse_natural_time, parse_to_iso


def get_session_id_from_runtime(runtime: ToolRuntime) -> str:
    """Return session_id from state or from configurable.thread_id."""
    state = runtime.state
    state_session_id = state.get("session_id")
    if state_session_id:
        return state_session_id
    config = getattr(runtime, "config", None) or {}
    return (config.get("configurable") or {}).get("thread_id") or "default"


@tool
def set_field_value(field_name: str, value: str, runtime: ToolRuntime) -> Command:
    """
    Set one required field from the user's message. Call only when status is 'collecting'.
    field_name must be exactly the next_required_field from the last tool response.
    value: value extracted from the user (for camera_id use the user's choice—name or ID; for time_window use JSON e.g. {"start": "08:00", "end": "10:00"} or a short description).
    """
    state = runtime.state
    if state.get("status") != "collecting":
        content = json.dumps(
            tool_response(
                status=state.get("status", "idle"),
                error="Cannot set field: status is not 'collecting'. Initialize a rule first.",
            )
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    missing_fields = state.get("missing_fields") or []
    if field_name not in missing_fields:
        content = json.dumps(
            tool_response(
                status="collecting",
                missing_fields=missing_fields,
                error=f"Field '{field_name}' is not in missing_fields. Only set one of: {missing_fields}. Use next_required_field.",
            )
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    rule = get_rule(state["rule_id"])
    if not rule:
        content = json.dumps(
            tool_response(status="collecting", error="Rule not found.")
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    # Resolve camera_id: user may provide camera name or ID; we store the resolved ID
    if field_name == "camera_id":
        user_id = state.get("user_id")
        session_id = get_session_id_from_runtime(runtime)
        if not user_id:
            content = json.dumps(
                tool_response(
                    status="collecting",
                    missing_fields=missing_fields,
                    error="Cannot resolve camera: user_id is missing. Camera selection requires an authenticated user.",
                )
            )
            return Command(
                update={
                    "messages": [
                        ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                    ]
                }
            )
        raw_value = (value or "").strip()
        if not raw_value:
            content = json.dumps(
                tool_response(
                    status="collecting",
                    missing_fields=missing_fields,
                    error="Camera choice is empty. Ask the user to pick a camera from the list by name or ID.",
                )
            )
            return Command(
                update={
                    "messages": [
                        ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                    ]
                }
            )
        resolved = resolve_camera_provider(
            name_or_id=raw_value, user_id=user_id, session_id=session_id
        )
        res_status = resolved.get("status")
        if res_status == "exact_match":
            value = resolved.get("camera_id") or raw_value
        elif res_status == "multiple_matches":
            cameras = resolved.get("cameras") or []
            names = [f"{camera.get('name', '')} (ID: {camera.get('id', '')})" for camera in cameras]
            content = json.dumps(
                tool_response(
                    status="collecting",
                    missing_fields=missing_fields,
                    error=f"Multiple cameras match. Ask the user to choose one: {names}.",
                )
            )
            return Command(
                update={
                    "messages": [
                        ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                    ]
                }
            )
        else:
            error_message = resolved.get("error") or "Camera not found."
            content = json.dumps(
                tool_response(
                    status="collecting",
                    missing_fields=missing_fields,
                    error=f"{error_message} Ask the user to choose a camera from the list by name or ID.",
                )
            )
            return Command(
                update={
                    "messages": [
                        ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                    ]
                }
            )

    try:
        parsed = (
            json.loads(value)
            if isinstance(value, str) and value.strip().startswith("{")
            else value
        )
    except json.JSONDecodeError:
        parsed = value

    # Normalize start_time/end_time: parse natural language to ISO before storing
    if field_name in ("start_time", "end_time"):
        raw_str = (parsed if isinstance(parsed, str) else str(parsed)).strip()
        if not raw_str:
            content = json.dumps(
                tool_response(
                    status="collecting",
                    missing_fields=missing_fields,
                    error=f"Empty {field_name}. Ask for a valid time (e.g. 'now', 'in 10 min', 'tomorrow 3pm').",
                )
            )
            return Command(
                update={"messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]}
            )
        try:
            ref = None
            if field_name == "end_time":
                start_val = (state.get("config") or {}).get("start_time")
                if start_val:
                    try:
                        if isinstance(start_val, datetime):
                            ref = start_val
                        else:
                            ref = parse_natural_time(str(start_val))
                    except ValueError:
                        pass
            parsed = parse_to_iso(raw_str, reference_datetime=ref)
        except ValueError as e:
            content = json.dumps(
                tool_response(
                    status="collecting",
                    missing_fields=missing_fields,
                    error=f"Invalid time '{raw_str}': {e}. Try phrases like 'now', 'in 10 min', 'tomorrow 3pm' or ISO format.",
                )
            )
            return Command(
                update={"messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]}
            )

    config = dict(state.get("config") or {})
    config[field_name] = parsed
    missing_fields = compute_missing_fields(rule, config)
    status = "confirmation" if not missing_fields else "collecting"
    next_required_field = get_next_required_field(missing_fields)

    cameras: list[dict[str, str]] = []
    user_id = state.get("user_id")
    session_id = get_session_id_from_runtime(runtime)
    if next_required_field == "camera_id" and user_id:
        cameras = get_cameras(user_id=user_id, session_id=session_id)

    if next_required_field in ("start_time", "end_time"):
        time_hint = get_current_time_format_hint()
        next_msg = f"Next: {next_required_field}. {time_hint}"
    elif next_required_field == "camera_id" and cameras:
        next_msg = (
            "Next required field is camera_id. Show the user the 'cameras' list below and ask them to choose one by name or ID. "
            "When they reply, call set_field_value(field_name='camera_id', value=<their choice>)."
        )
    elif next_required_field:
        next_msg = f"Next ask for: {next_required_field}. If the user gave more values in the same message, you may call set_field_value again for the next field."
    else:
        next_msg = "All fields collected. Call save_to_db(); human approval will be requested by the system."
    content = json.dumps(
        tool_response(
            status=status,
            next_required_field=next_required_field,
            missing_fields=missing_fields,
            message=f"Set '{field_name}'. " + next_msg,
            config_preview=config if status == "confirmation" else None,
            cameras=cameras if cameras else None,
        )
    )
    return Command(
        update={
            "config": config,
            "missing_fields": missing_fields,
            "status": status,
            "messages": [
                ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
            ],
        }
    )
