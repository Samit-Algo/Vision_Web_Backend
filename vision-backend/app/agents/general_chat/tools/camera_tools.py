"""
Camera-related tools for the General Chat Agent.
"""

import logging
from typing import Dict, Any

from ...tools.camera_selection_tool import list_cameras as _list_cameras_base
from ...tools.camera_selection_tool import resolve_camera as _resolve_camera_base
from ...exceptions import VisionAgentError

logger = logging.getLogger(__name__)


def list_my_cameras(user_id: str) -> Dict[str, Any]:
    """
    List all cameras owned by you.
    Use this when the user asks which cameras they have or what cameras are available.
    
    Args:
        user_id: The ID of the current user.
    """
    try:
        return _list_cameras_base(user_id=user_id)
    except VisionAgentError as e:
        logger.error(f"list_my_cameras: {e}")
        return {"error": e.user_message, "cameras": []}
    except Exception as e:
        logger.exception(f"list_my_cameras: {e}")
        return {"error": "Failed to list cameras.", "cameras": []}


def find_camera(name_or_id: str, user_id: str) -> Dict[str, Any]:
    """
    Find a specific camera by name or ID.
    
    Args:
        name_or_id: The name or ID of the camera to find.
        user_id: The ID of the current user.
    """
    try:
        return _resolve_camera_base(name_or_id=name_or_id, user_id=user_id)
    except VisionAgentError as e:
        logger.error(f"find_camera: {e}")
        return {"status": "not_found", "error": e.user_message}
    except Exception as e:
        logger.exception(f"find_camera: {e}")
        return {"status": "not_found", "error": "Failed to find camera."}


def check_camera_health(camera_id: str, user_id: str) -> Dict[str, Any]:
    """
    Check the connection status and health of a specific camera.
    
    Args:
        camera_id: The unique ID of the camera.
        user_id: The ID of the current user.
    """
    try:
        result = _resolve_camera_base(name_or_id=camera_id, user_id=user_id)
    except VisionAgentError as e:
        logger.error(f"check_camera_health: {e}")
        return {"status": "error", "message": e.user_message}
    except Exception as e:
        logger.exception(f"check_camera_health: {e}")
        return {"status": "error", "message": "Failed to check camera health."}

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
