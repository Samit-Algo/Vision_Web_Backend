"""
Structured tool response for the LLM. Backend controls what the LLM sees.
"""

from datetime import datetime, timezone
from typing import Any


def get_current_time_format_hint() -> str:
    """Hint for LLM: natural language time is accepted. Pass user's exact words to set_field_value."""
    now = datetime.now(timezone.utc)
    example = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return (
        f"Current time (UTC): {example}. "
        "Natural language is accepted: 'now', 'in 10 min', 'after 10 min', 'tomorrow 3pm', 'tomorrow 7', "
        "or ISO (e.g. 2025-02-20T08:00:00+00:00). Pass the user's exact words to set_field_value for each field."
    )


def tool_response(
    *,
    status: str,
    next_required_field: str | None = None,
    missing_fields: list[str] | None = None,
    error: str | None = None,
    message: str | None = None,
    final_payload: dict | None = None,
    config_preview: dict | None = None,
    cameras: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build the JSON object returned in tool messages."""
    out: dict[str, Any] = {"status": status}
    if next_required_field is not None:
        out["next_required_field"] = next_required_field
    if missing_fields is not None:
        out["missing_fields"] = missing_fields
    if error:
        out["error"] = error
    if message:
        out["message"] = message
    if final_payload is not None:
        out["final_payload"] = final_payload
    if config_preview is not None:
        out["config_preview"] = config_preview
    if cameras is not None:
        out["cameras"] = cameras
    return out
