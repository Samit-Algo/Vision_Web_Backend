"""
Tool: reopen_for_editing(). Set status back to collecting so the user can change a field. Allowed only when status is confirmation (e.g. after user rejected save).
"""

import json

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..state import get_editable_fields, get_rule
from .response import tool_response


@tool
def reopen_for_editing(runtime: ToolRuntime) -> Command:
    """
    Reopen the configuration for editing. Call when the user rejected the save and wants to change something. Sets status back to 'collecting' and exposes editable fields so you can ask which field they want to change and call set_field_value.
    """
    state = runtime.state
    if state.get("status") != "confirmation":
        content = json.dumps(
            tool_response(
                status=state.get("status", "idle"),
                error="Cannot reopen: status is not 'confirmation'. Only use after user rejected the save.",
            )
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    rule = get_rule(state.get("rule_id") or "")
    if not rule:
        content = json.dumps(
            tool_response(status="confirmation", error="Rule not found.")
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    config = state.get("config") or {}
    missing_fields = get_editable_fields(rule, config)
    next_required_field = missing_fields[0] if missing_fields else None

    content = json.dumps(
        tool_response(
            status="collecting",
            next_required_field=next_required_field,
            missing_fields=missing_fields,
            message="Configuration reopened for editing. Ask the user which field they want to change, then call set_field_value for that field.",
        )
    )
    return Command(
        update={
            "status": "collecting",
            "missing_fields": missing_fields,
            "messages": [
                ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
            ],
        }
    )
