from __future__ import annotations

import asyncio
from typing import Any, Dict

import nest_asyncio

from ...application.use_cases.camera.list_cameras import ListCamerasUseCase
from ..session_state.agent_state import get_agent_state
from .save_to_db_tool import (
    get_camera_repository as _get_camera_repository_shared,
    set_camera_repository as _set_camera_repository_shared,
)

nest_asyncio.apply()


# ============================================================================
# REPOSITORY MANAGEMENT
# ============================================================================

def set_camera_repository(repository):
    """Set the camera repository for camera operations."""
    _set_camera_repository_shared(repository)
    print(f"[camera_selection_tool.set_camera_repository] Camera repository set: {type(repository)}")


def _get_camera_repository():
    """Get the shared camera repository from save_to_db_tool."""
    return _get_camera_repository_shared()


# ============================================================================
# CAMERA OPERATIONS
# ============================================================================

def list_cameras(user_id: str, session_id: str = "default") -> Dict[str, Any]:
    """
    List all cameras owned by a user.

    Args:
        user_id: The user ID to list cameras for
        session_id: Session identifier (for consistency with other tools)

    Returns:
        Dict with cameras list:
        {
            "cameras": [
                {"id": "CAM-001", "name": "Front Gate"},
                {"id": "CAM-002", "name": "Warehouse Cam 2"}
            ]
        }
    """
    if not user_id:
        return {
            "error": "user_id is required",
            "cameras": []
        }

    camera_repository = _get_camera_repository()
    if not camera_repository:
        return {
            "error": "Camera repository not initialized. Call set_camera_repository() first.",
            "cameras": []
        }

    try:
        list_cameras_use_case = ListCamerasUseCase(camera_repository)
        camera_responses = asyncio.run(list_cameras_use_case.execute(user_id))

        cameras = []
        for camera_response in camera_responses:
            cameras.append({
                "id": camera_response.id,
                "name": camera_response.name
            })

        print(f"[list_cameras] Found {len(cameras)} cameras for user {user_id}")
        return {
            "cameras": cameras
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[list_cameras] ERROR: Failed to list cameras for user {user_id}: {str(e)}")
        print(f"[list_cameras] Traceback: {error_trace}")
        return {
            "error": f"Failed to list cameras: {str(e)}",
            "cameras": []
        }


def resolve_camera(name_or_id: str, user_id: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Resolve a camera by name (partial match) or ID.

    Args:
        name_or_id: Camera name (partial match supported) or camera ID
        user_id: The user ID to filter cameras by
        session_id: Session identifier (for consistency with other tools)

    Returns:
        Dict with status:
        {
            "status": "exact_match",
            "camera_id": "CAM-001",
            "camera_name": "Front Gate"
        }
        OR
        {
            "status": "multiple_matches",
            "cameras": [...]
        }
        OR
        {
            "status": "not_found"
        }
    """
    if not name_or_id or not user_id:
        return {
            "status": "not_found",
            "error": "name_or_id and user_id are required"
        }

    camera_repository = _get_camera_repository()
    if not camera_repository:
        return {
            "status": "not_found",
            "error": "Camera repository not initialized. Call set_camera_repository() first."
        }

    try:
        camera_by_id = asyncio.run(camera_repository.find_by_id(name_or_id))

        if camera_by_id:
            if camera_by_id.owner_user_id == user_id:
                print(f"[resolve_camera] Found exact match by ID: {camera_by_id.id} ({camera_by_id.name})")
                return {
                    "status": "exact_match",
                    "camera_id": camera_by_id.id or "",
                    "camera_name": camera_by_id.name
                }
            else:
                print(f"[resolve_camera] Camera {camera_by_id.id} found but belongs to different user")

        print(f"[resolve_camera] Searching by name: '{name_or_id}' for user {user_id}")
        cameras_by_name = asyncio.run(
            camera_repository.search_by_name(name_or_id, user_id, limit=10)
        )

        cameras_by_name = [c for c in cameras_by_name if c.owner_user_id == user_id]

        print(f"[resolve_camera] Found {len(cameras_by_name)} cameras matching '{name_or_id}'")

        if len(cameras_by_name) == 0:
            return {
                "status": "not_found"
            }
        elif len(cameras_by_name) == 1:
            camera = cameras_by_name[0]
            print(f"[resolve_camera] Found exact match by name: {camera.id} ({camera.name})")
            return {
                "status": "exact_match",
                "camera_id": camera.id or "",
                "camera_name": camera.name
            }
        else:
            cameras_list = []
            for camera in cameras_by_name:
                cameras_list.append({
                    "id": camera.id or "",
                    "name": camera.name
                })
            print(f"[resolve_camera] Found {len(cameras_list)} multiple matches")
            return {
                "status": "multiple_matches",
                "cameras": cameras_list
            }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[resolve_camera] ERROR: Failed to resolve camera '{name_or_id}' for user {user_id}: {str(e)}")
        print(f"[resolve_camera] Traceback: {error_trace}")
        return {
            "status": "not_found",
            "error": f"Failed to resolve camera: {str(e)}"
        }
