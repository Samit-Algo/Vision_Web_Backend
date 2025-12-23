"""Event Controller

Controller to receive and store events from Jetson devices.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
import logging

from ...utils.event_storage import save_event_from_payload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["events"])


@router.post("/events", status_code=status.HTTP_201_CREATED)
async def receive_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Receive event notification from Jetson device.
    
    Expected payload structure:
    {
        "event": {
            "label": "Person detected",
            "rule_index": 0,
            "timestamp": "2025-01-15T10:30:45Z"
        },
        "agent": {
            "agent_id": "agent-123",
            "agent_name": "Security Agent",
            "camera_id": "camera-456"
        },
        "frame": {
            "image_base64": "...",
            "format": "jpeg"
        },
        "metadata": {
            "video_timestamp": "0:01:23.456",
            "detections": {...}
        }
    }
    
    Args:
        payload: Event payload from Jetson device
        
    Returns:
        Success response with saved file paths
    """
    try:
        # Validate payload
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty payload"
            )
        
        agent_info = payload.get("agent", {})
        camera_id = agent_info.get("camera_id")
        agent_id = agent_info.get("agent_id")
        
        if not camera_id or not agent_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing camera_id or agent_id in payload"
            )
        
        # Save event to local files
        saved_paths = save_event_from_payload(payload)
        
        logger.info(
            f"Event received and saved: camera_id={camera_id}, "
            f"agent_id={agent_id}, paths={saved_paths}"
        )
        
        return {
            "status": "success",
            "message": "Event saved successfully",
            "paths": saved_paths
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing event: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process event: {str(e)}"
        )


@router.post("/agents/{agent_id}/events", status_code=status.HTTP_201_CREATED)
async def receive_event_by_agent(agent_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Receive event notification with agent_id in path.
    The agent_id in path will override agent_id in payload if different.
    """
    # Override agent_id in payload if provided in path
    if "agent" not in payload:
        payload["agent"] = {}
    payload["agent"]["agent_id"] = agent_id
    
    return await receive_event(payload)

