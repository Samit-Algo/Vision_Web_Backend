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


def get_video_storage_path(session_id: str) -> Path:
    """
    Get the storage path for video chunks for a specific session.
    
    Structure: events/videos/{session_id}/
    
    Args:
        session_id: The session identifier (format: agent_id_rule_index_timestamp)
        
    Returns:
        Path object for the storage directory
    """
    storage_path = EVENTS_BASE_DIR / "videos" / session_id
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path


def save_video_chunk_from_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Save video chunk from the payload structure sent by Jetson device.
    
    Expected payload structure (event_video type):
    {
        "type": "event_video",
        "session_id": "agent123_0_1704110400",
        "sequence_number": 0,
        "is_final_chunk": false,
        "chunk": {
            "chunk_number": 0,
            "start_time": "2024-01-01T12:00:00Z",
            "end_time": "2024-01-01T12:05:00Z",
            "duration_seconds": 300.0
        },
        "event": {
            "label": "person without helmet",
            "rule_index": 0,
            "timestamp": "2024-01-01T12:00:00Z"
        },
        "agent": {
            "agent_id": "agent123",
            "agent_name": "Helmet Detection",
            "camera_id": "cam001"
        },
        "camera": {
            "owner_user_id": "user123",
            "device_id": "device456"
        },
        "video": {
            "data_base64": "...",
            "format": "mp4",
            "fps": 5,
            "resolution": {
                "width": 1280,
                "height": 720
            }
        },
        "metadata": {
            "session_id": "agent123_0_1704110400",
            "chunk_sequence": 0
        }
    }
    
    Args:
        payload: Video chunk payload from Jetson device
        
    Returns:
        Dictionary with saved file paths:
        {
            "video_path": "events/videos/session_id/chunk_0.mp4",
            "metadata_path": "events/videos/session_id/chunk_0.json"
        }
    """
    try:
        # Extract session_id
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("Missing session_id in video chunk payload")
        
        # Extract chunk information
        chunk_info = payload.get("chunk", {})
        chunk_number = chunk_info.get("chunk_number", payload.get("sequence_number", 0))
        
        # Get storage path for this session
        storage_path = get_video_storage_path(session_id)
        
        # Decode and save video
        video_info = payload.get("video", {})
        video_base64 = video_info.get("data_base64")
        
        if not video_base64:
            raise ValueError("Missing video data_base64 in payload")
        
        # Decode base64 video
        video_data = base64.b64decode(video_base64)
        
        # Save video file
        video_filename = f"chunk_{chunk_number}.mp4"
        video_path = storage_path / video_filename
        with open(video_path, "wb") as f:
            f.write(video_data)
        
        # Prepare metadata (everything except the video data)
        metadata = {
            "session_id": session_id,
            "chunk_number": chunk_number,
            "sequence_number": payload.get("sequence_number", chunk_number),
            "is_final_chunk": payload.get("is_final_chunk", False),
            "chunk": chunk_info,
            "event": payload.get("event", {}),
            "agent": payload.get("agent", {}),
            "camera": payload.get("camera", {}),
            "video_metadata": {
                "format": video_info.get("format", "mp4"),
                "fps": video_info.get("fps"),
                "resolution": video_info.get("resolution", {})
            },
            "metadata": payload.get("metadata", {}),
            "received_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Save metadata as JSON
        metadata_filename = f"chunk_{chunk_number}.json"
        metadata_path = storage_path / metadata_filename
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(
            f"Saved video chunk: session_id={session_id}, "
            f"chunk_number={chunk_number}, video_path={video_path}"
        )
        
        return {
            "video_path": str(video_path),
            "metadata_path": str(metadata_path),
            "session_id": session_id,
            "chunk_number": chunk_number
        }
        
    except Exception as e:
        logger.error(f"Failed to save video chunk from payload: {e}")
        raise


def get_video_chunk_path(session_id: str, chunk_number: int) -> Optional[Path]:
    """
    Get the file path for a video chunk.
    
    Args:
        session_id: The session identifier
        chunk_number: The chunk number
        
    Returns:
        Path to video file if exists, None otherwise
    """
    storage_path = get_video_storage_path(session_id)
    video_path = storage_path / f"chunk_{chunk_number}.mp4"
    
    if video_path.exists():
        return video_path
    return None


def get_video_chunk_metadata_path(session_id: str, chunk_number: int) -> Optional[Path]:
    """
    Get the file path for a video chunk metadata.
    
    Args:
        session_id: The session identifier
        chunk_number: The chunk number
        
    Returns:
        Path to metadata file if exists, None otherwise
    """
    storage_path = get_video_storage_path(session_id)
    metadata_path = storage_path / f"chunk_{chunk_number}.json"
    
    if metadata_path.exists():
        return metadata_path
    return None


def list_video_chunks_for_session(session_id: str) -> list[Dict[str, Any]]:
    """
    List all video chunks for a session.
    
    Args:
        session_id: The session identifier
        
    Returns:
        List of dictionaries with chunk information:
        [
            {
                "chunk_number": 0,
                "video_path": "...",
                "metadata_path": "...",
                "metadata": {...}
            },
            ...
        ]
    """
    try:
        storage_path = get_video_storage_path(session_id)
        
        if not storage_path.exists():
            return []
        
        chunks = []
        
        # Find all chunk files
        for video_file in storage_path.glob("chunk_*.mp4"):
            chunk_number_str = video_file.stem.replace("chunk_", "")
            try:
                chunk_number = int(chunk_number_str)
                
                metadata_path = storage_path / f"chunk_{chunk_number}.json"
                metadata = {}
                
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                
                chunks.append({
                    "chunk_number": chunk_number,
                    "video_path": str(video_file),
                    "metadata_path": str(metadata_path) if metadata_path.exists() else None,
                    "metadata": metadata
                })
            except ValueError:
                continue
        
        # Sort by chunk number
        chunks.sort(key=lambda x: x["chunk_number"])
        
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to list video chunks for session {session_id}: {e}")
        return []

