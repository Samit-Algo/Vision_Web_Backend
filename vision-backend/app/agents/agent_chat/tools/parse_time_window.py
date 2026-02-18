from __future__ import annotations

from typing import Any, Dict, Optional

from ..state.agent_state import get_agent_state
from ..utils.time_parser import parse_time_window as parse_time_window_impl


def parse_time_window(
    user_time_phrase: str,
    reference_start_iso: Optional[str] = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    if not user_time_phrase or not str(user_time_phrase).strip():
        return {
            "success": False,
            "user_message": "Please specify when to start and when to end (e.g. 'now', then 'in 10 minutes').",
        }
    ref = reference_start_iso
    if not ref and session_id:
        try:
            state = get_agent_state(session_id)
            ref = state.fields.get("start_time")
            if not isinstance(ref, str) or not ref.strip():
                ref = None
        except Exception:
            ref = None
    try:
        result = parse_time_window_impl(str(user_time_phrase).strip(), reference_start_iso=ref)
    except Exception:
        return {
            "success": False,
            "user_message": "I couldn't understand that time. Try 'now' and 'in 10 minutes', or 'Sunday 1 to 5 PM'.",
        }
    if not result.get("success"):
        return result
    out = {
        "success": True,
        "start_time": result.get("start_time"),
        "end_time": result.get("end_time"),
        "user_message": result.get("user_message"),
    }
    if result.get("duration_minutes") is not None:
        out["duration_minutes"] = result["duration_minutes"]
    return out
