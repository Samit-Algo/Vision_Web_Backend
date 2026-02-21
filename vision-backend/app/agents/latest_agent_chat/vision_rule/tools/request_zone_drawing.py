"""
Tool: request_zone_drawing(). Ask the user to draw the zone in the UI canvas.
Human-in-the-loop: when the LLM calls this, execution is interrupted; the UI shows
the zone drawing canvas. When the user saves the drawing, the client sends a resume
with zone data; the backend injects it into config and the agent continues.
"""

import json
from typing import Any

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..state import compute_missing_fields, get_next_required_field, get_rule
from .response import tool_response


def get_zone_from_resume(runtime: ToolRuntime) -> Any:
    """Return zone data from configurable (injected by use case when resuming with zone)."""
    config = getattr(runtime, "config", None) or {}
    return (config.get("configurable") or {}).get("zone_data")


@tool
def request_zone_drawing(runtime: ToolRuntime) -> Command:
    """
    Request the user to draw the zone in the zone drawing canvas. Call this when
    next_required_field is 'zone'. Do not ask the user to type or describe the zone—
    the system will show a canvas; after the user draws and saves, the zone is set automatically.
    """
    state = runtime.state
    if state.get("status") != "collecting":
        content = json.dumps(
            tool_response(
                status=state.get("status", "idle"),
                error="Cannot request zone drawing: status is not 'collecting'. Initialize and collect other fields first.",
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
    if "zone" not in missing_fields:
        content = json.dumps(
            tool_response(
                status="collecting",
                missing_fields=missing_fields,
                error="Zone is not in missing_fields. Only call request_zone_drawing when next_required_field is 'zone'.",
            )
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    zone_data = get_zone_from_resume(runtime)
    if zone_data is not None:
        config = dict(state.get("config") or {})
        config["zone"] = zone_data
        rule = get_rule(state.get("rule_id") or "")
        if rule:
            missing_fields = compute_missing_fields(rule, config)
            status = "confirmation" if not missing_fields else "collecting"
            next_required_field = get_next_required_field(missing_fields)
            content = json.dumps(
                tool_response(
                    status=status,
                    next_required_field=next_required_field,
                    missing_fields=missing_fields,
                    message="Zone set from drawing. " + (
                        f"Next ask for: {next_required_field}." if next_required_field else "All fields collected. Call save_to_db(); human approval will be requested."
                    ),
                    config_preview=config if status == "confirmation" else None,
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

    # First call: no zone data yet — middleware will interrupt; return message for when tool runs after interrupt
    content = json.dumps(
        tool_response(
            status="collecting",
            next_required_field="zone",
            missing_fields=missing_fields,
            message="Waiting for user to draw the zone in the canvas. Do not ask in chat—the UI will show the zone editor.",
        )
    )
    return Command(
        update={
            "messages": [
                ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
            ]
        }
    )
