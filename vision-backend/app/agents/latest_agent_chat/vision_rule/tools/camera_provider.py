"""
Camera list and resolve for vision-rule agent.
Uses main app's camera_selection_tool when available; otherwise returns empty/not_found.
"""

from __future__ import annotations

from typing import Any

_list_cameras_fn = None
_resolve_camera_fn = None

try:
    from app.agents.tools.camera_selection_tool import (
        list_cameras as _list_cameras_fn,
        resolve_camera as _resolve_camera_fn,
    )
except ImportError:
    pass


def get_cameras(user_id: str, session_id: str = "default") -> list[dict[str, str]]:
    """
    List cameras for the user. Returns [{"id": "...", "name": "..."}, ...].
    Returns [] if user_id is missing or if camera service is unavailable.
    """
    if not user_id:
        return []
    if _list_cameras_fn is None:
        return []
    try:
        result = _list_cameras_fn(user_id=user_id, session_id=session_id)
        return result.get("cameras") or []
    except Exception:
        return []


def resolve_camera(
    name_or_id: str, user_id: str, session_id: str = "default"
) -> dict[str, Any]:
    """
    Resolve a camera by name (partial match) or ID.
    Returns dict with:
      - status: "exact_match" | "multiple_matches" | "not_found"
      - camera_id, camera_name when exact_match
      - cameras: list when multiple_matches
      - error: str when validation/service error
    """
    if not name_or_id or not user_id:
        return {"status": "not_found", "error": "Missing name_or_id or user_id"}
    if _resolve_camera_fn is None:
        return {"status": "not_found", "error": "Camera service not available"}
    try:
        return _resolve_camera_fn(
            name_or_id=name_or_id.strip(),
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as e:
        return {"status": "not_found", "error": str(e)}
