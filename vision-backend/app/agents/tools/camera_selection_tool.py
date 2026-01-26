from __future__ import annotations

from typing import Dict, Any, Optional, List
import asyncio

from ..session_state.agent_state import get_agent_state
from ...application.use_cases.camera.list_cameras import ListCamerasUseCase

# Enable nested event loops to allow asyncio.run() from within async contexts
import nest_asyncio
nest_asyncio.apply()


# Import the shared camera repository functions from save_to_db_tool
# This ensures both tools use the same repository instance
from .save_to_db_tool import set_camera_repository as _set_camera_repository_shared, get_camera_repository as _get_camera_repository_shared


def set_camera_repository(repository):
    """
    Set the camera repository for camera operations.
    This delegates to the shared repository in save_to_db_tool so both tools use the same instance.
    """
    _set_camera_repository_shared(repository)
    print(f"[camera_selection_tool.set_camera_repository] Camera repository set: {type(repository)}")


def _get_camera_repository():
    """
    Get the shared camera repository from save_to_db_tool.
    This ensures we use the same repository instance that was set via dependency injection.
    """
    return _get_camera_repository_shared()


def list_cameras(user_id: str, session_id: str = "default") -> Dict[str, Any]:
    """
    List all cameras owned by a user.
    
    This function reuses the existing ListCamerasUseCase to get all cameras for a user.
    Use this to suggest available cameras to the user when camera_id is missing.
    
    Args:
        user_id (str): The user ID to list cameras for
        session_id (str): Session identifier (for consistency with other tools)
    
    Returns:
        Dict: A dictionary with cameras list:
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
        # Reuse existing ListCamerasUseCase
        list_cameras_use_case = ListCamerasUseCase(camera_repository)
        camera_responses = asyncio.run(list_cameras_use_case.execute(user_id))
        
        # Convert to simple format for LLM
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
    Resolve a camera by name (regex/partial match) or ID.
    
    This function:
    1. First tries to find by ID (exact match, fast)
    2. If not found, searches by name using regex (case-insensitive partial match)
    3. Returns structured response:
       - exact_match: Single camera found
       - multiple_matches: Multiple cameras found (LLM should ask user to clarify)
       - not_found: No cameras found
    
    Args:
        name_or_id (str): Camera name (partial match supported) or camera ID
        user_id (str): The user ID to filter cameras by
        session_id (str): Session identifier (for consistency with other tools)
    
    Returns:
        Dict: Structured response with status:
        {
            "status": "exact_match",
            "camera_id": "CAM-001",
            "camera_name": "Front Gate"
        }
        OR
        {
            "status": "multiple_matches",
            "cameras": [
                {"id": "CAM-001", "name": "Loading Area"},
                {"id": "CAM-002", "name": "Loading Dock"}
            ]
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
        # Step 1: Try find_by_id first (cheap + exact)
        camera_by_id = asyncio.run(camera_repository.find_by_id(name_or_id))
        
        if camera_by_id:
            # Verify it belongs to the user
            if camera_by_id.owner_user_id == user_id:
                print(f"[resolve_camera] Found exact match by ID: {camera_by_id.id} ({camera_by_id.name})")
                return {
                    "status": "exact_match",
                    "camera_id": camera_by_id.id or "",
                    "camera_name": camera_by_id.name
                }
            else:
                print(f"[resolve_camera] Camera {camera_by_id.id} found but belongs to different user")
        
        # Step 2: If not found by ID, search by name (regex)
        print(f"[resolve_camera] Searching by name: '{name_or_id}' for user {user_id}")
        cameras_by_name = asyncio.run(
            camera_repository.search_by_name(name_or_id, user_id, limit=10)
        )
        
        # Filter to ensure they belong to the user (safety check)
        cameras_by_name = [c for c in cameras_by_name if c.owner_user_id == user_id]
        
        print(f"[resolve_camera] Found {len(cameras_by_name)} cameras matching '{name_or_id}'")
        
        if len(cameras_by_name) == 0:
            return {
                "status": "not_found"
            }
        elif len(cameras_by_name) == 1:
            # Single match - exact match
            camera = cameras_by_name[0]
            print(f"[resolve_camera] Found exact match by name: {camera.id} ({camera.name})")
            return {
                "status": "exact_match",
                "camera_id": camera.id or "",
                "camera_name": camera.name
            }
        else:
            # Multiple matches - return list for LLM to handle
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
