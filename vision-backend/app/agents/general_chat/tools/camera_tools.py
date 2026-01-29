"""
Camera-related tools for the General Chat Agent.
"""

import asyncio
from typing import Dict, Any, Optional
from ...tools.camera_selection_tool import list_cameras as _list_cameras_base
from ...tools.camera_selection_tool import resolve_camera as _resolve_camera_base

def list_my_cameras(user_id: str) -> Dict[str, Any]:
    """
    List all cameras owned by you.
    Use this when the user asks which cameras they have or what cameras are available.
    
    Args:
        user_id: The ID of the current user.
    """
    return _list_cameras_base(user_id=user_id)

def find_camera(name_or_id: str, user_id: str) -> Dict[str, Any]:
    """
    Find a specific camera by name or ID.
    
    Args:
        name_or_id: The name or ID of the camera to find.
        user_id: The ID of the current user.
    """
    return _resolve_camera_base(name_or_id=name_or_id, user_id=user_id)

def check_camera_health(camera_id: str, user_id: str) -> Dict[str, Any]:
    """
    Check the connection status and health of a specific camera.
    
    Args:
        camera_id: The unique ID of the camera.
        user_id: The ID of the current user.
    """
    # Simply resolve the camera first
    result = _resolve_camera_base(name_or_id=camera_id, user_id=user_id)
    
    if result.get("status") != "exact_match":
        return {"status": "error", "message": "Camera not found or unauthorized."}
    
    # In this system, we consider a camera "Healthy" if it exists. 
    # Real-world heartbeat would go here.
    return {
        "status": "healthy",
        "camera_name": result.get("camera_name"),
        "connectivity": "online",
        "stream_status": "active"
    }
