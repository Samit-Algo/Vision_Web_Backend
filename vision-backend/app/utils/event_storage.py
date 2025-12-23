"""Event Storage Utility

Utility to save event data and annotated frames to local files,
organized by camera_id and agent_id.
"""

import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Base directory for storing events
EVENTS_BASE_DIR = Path("events")


def get_event_storage_path(camera_id: str, agent_id: str) -> Path:
    """
    Get the storage path for events for a specific camera and agent.
    
    Structure: events/{camera_id}/{agent_id}/
    
    Args:
        camera_id: The camera identifier
        agent_id: The agent identifier
        
    Returns:
        Path object for the storage directory
    """
    storage_path = EVENTS_BASE_DIR / camera_id / agent_id
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path


def save_event_to_file(
    camera_id: str,
    agent_id: str,
    event_data: Dict[str, Any],
    frame_base64: Optional[str] = None
) -> Dict[str, str]:
    """
    Save event data and frame to local files.
    
    Creates the directory structure: events/{camera_id}/{agent_id}/
    Saves:
    - {timestamp}.json - Event metadata
    - {timestamp}.jpg - Annotated frame (if provided)
    
    Args:
        camera_id: The camera identifier
        agent_id: The agent identifier
        event_data: Dictionary containing event information
        frame_base64: Base64-encoded JPEG image (optional)
        
    Returns:
        Dictionary with saved file paths:
        {
            "json_path": "events/camera_id/agent_id/timestamp.json",
            "image_path": "events/camera_id/agent_id/timestamp.jpg" (if frame provided)
        }
    """
    try:
        # Get storage path and ensure it exists
        storage_path = get_event_storage_path(camera_id, agent_id)
        
        # Generate timestamp for filenames
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save event metadata as JSON
        json_path = storage_path / f"{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(event_data, f, indent=2, ensure_ascii=False)
        
        result = {
            "json_path": str(json_path),
        }
        
        # Save frame if provided
        if frame_base64:
            try:
                # Decode base64 image
                image_data = base64.b64decode(frame_base64)
                
                # Save as JPEG
                image_path = storage_path / f"{timestamp}.jpg"
                with open(image_path, "wb") as f:
                    f.write(image_data)
                
                result["image_path"] = str(image_path)
                logger.info(f"Saved event frame: {image_path}")
            except Exception as e:
                logger.error(f"Failed to save event frame: {e}")
                result["image_error"] = str(e)
        
        logger.info(f"Saved event data: {json_path}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to save event to file: {e}")
        raise


def save_event_from_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Save event from the payload structure sent by Jetson device.
    
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
        Dictionary with saved file paths
    """
    try:
        # Extract required information
        agent_info = payload.get("agent", {})
        camera_id = agent_info.get("camera_id")
        agent_id = agent_info.get("agent_id")
        
        if not camera_id or not agent_id:
            raise ValueError("Missing camera_id or agent_id in payload")
        
        # Extract frame if present
        frame_info = payload.get("frame", {})
        frame_base64 = frame_info.get("image_base64")
        
        # Prepare event data (everything except the frame)
        event_data = {
            "event": payload.get("event", {}),
            "agent": agent_info,
            "metadata": payload.get("metadata", {}),
            "received_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Save to files
        return save_event_to_file(
            camera_id=camera_id,
            agent_id=agent_id,
            event_data=event_data,
            frame_base64=frame_base64
        )
        
    except Exception as e:
        logger.error(f"Failed to save event from payload: {e}")
        raise

