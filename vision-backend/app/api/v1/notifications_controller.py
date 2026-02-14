"""
Notifications API: WebSocket for real-time events, video chunks, event images, push (FCM).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...application.dto.user_dto import UserResponse
from ...core.security import decode_jwt_token
from ...infrastructure.notifications import (
    WebSocketManager,
    send_push_notification,
    send_push_notification_multicast,
)
from ...utils.event_storage import (
    EVENTS_BASE_DIR,
    get_video_chunk_metadata_path,
    get_video_chunk_path,
    list_video_chunks_for_session,
)

from .dependencies import get_current_user

# -----------------------------------------------------------------------------
# Logging and router
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
router = APIRouter(tags=["notifications"])

# Set from main.py at startup
global_websocket_manager: Optional[WebSocketManager] = None
registered_device_tokens: List[str] = []


def set_websocket_manager(manager: WebSocketManager) -> None:
    """Set the global WebSocket manager (called from main.py lifespan)."""
    global global_websocket_manager
    global_websocket_manager = manager


def get_websocket_manager() -> WebSocketManager:
    """Return the global WebSocket manager. Raises if not initialized."""
    if global_websocket_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WebSocket manager not available",
        )
    return global_websocket_manager


async def get_user_id_from_token(token: str) -> str:
    """
    Extract user ID from JWT token.
    
    Args:
        token: JWT access token
        
    Returns:
        User ID from token
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        payload = decode_jwt_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise ValueError("Token does not contain user ID")
        return user_id
    except Exception as e:
        logger.warning("Invalid token for WebSocket connection: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


@router.websocket("/ws")
async def websocket_notifications(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT access token for authentication"),
):
    """
    WebSocket endpoint for receiving real-time event notifications.
    
    Clients connect to this endpoint to receive notifications when events
    are detected by the Jetson backend and sent via Kafka.
    
    Authentication is required via JWT token passed as query parameter.
    
    Example connection:
        ws://host/api/v1/notifications/ws?token=<jwt_token>
    
    Args:
        websocket: WebSocket connection instance
        token: JWT access token for authentication (query parameter)
    """
    # Get WebSocket manager from global instance
    try:
        manager = get_websocket_manager()
    except HTTPException as e:
        await websocket.close(code=1008, reason=str(e.detail))
        return
    
    if not token:
        await websocket.close(code=1008, reason="Authentication token required")
        return
    
    # Authenticate user
    try:
        user_id = await get_user_id_from_token(token)
    except HTTPException:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return
    
    # Accept WebSocket connection
    await websocket.accept()
    logger.info("WebSocket connection accepted for user %s", user_id)
    
    try:
        # Register connection with manager
        await manager.add_connection(user_id, websocket)
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to notifications service",
            "user_id": user_id
        })
        
        # Keep connection alive and handle incoming messages (ping/pong)
        while True:
            try:
                # Wait for any message from client (ping, pong, or other)
                message = await websocket.receive_text()
                
                # Handle ping messages (keep-alive)
                if message == "ping":
                    await websocket.send_text("pong")
                elif message == "pong":
                    # Client responded to our ping
                    pass
                else:
                    # Log unexpected messages
                    logger.debug("Received message from user %s: %s", user_id, message)
                    
            except WebSocketDisconnect:
                # Client disconnected
                logger.info("WebSocket disconnected for user %s", user_id)
                break
            except Exception as e:
                logger.error("Error handling WebSocket message for user %s: %s", user_id, e, exc_info=True)
                break
                
    except Exception as e:
        logger.error("Error in WebSocket connection for user %s: %s", user_id, e, exc_info=True)
    finally:
        # Clean up: remove connection from manager
        try:
            await manager.remove_connection(user_id, websocket)
            logger.info("WebSocket connection cleaned up for user %s", user_id)
        except Exception as e:
            logger.error("Error cleaning up WebSocket connection for user %s: %s", user_id, e, exc_info=True)


@router.get("/video-chunks/{session_id}")
async def list_video_chunks(
    session_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    List all video chunks for a specific session.
    
    Returns metadata about all available video chunks for the given session_id.
    This allows the frontend to know which chunks are available before requesting them.
    
    Args:
        session_id: The session identifier (format: agent_id_rule_index_timestamp)
        current_user: Current authenticated user (from dependency)
        
    Returns:
        Dictionary containing:
        {
            "session_id": "...",
            "chunks": [
                {
                    "chunk_number": 0,
                    "metadata": {...},
                    "video_path": "...",
                    "metadata_path": "..."
                },
                ...
            ]
        }
        
    Raises:
        HTTPException: If session not found or access denied
    """
    try:
        chunks = list_video_chunks_for_session(session_id)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No video chunks found for session",
            )
        
        return {
            "session_id": session_id,
            "chunks": chunks
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing video chunks for session %s: %s", session_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error listing video chunks",
        )


@router.get("/video-chunks/{session_id}/{chunk_number}")
async def get_video_chunk(
    session_id: str,
    chunk_number: int,
    current_user: UserResponse = Depends(get_current_user),
) -> FileResponse:
    """
    Get a specific video chunk by session_id and chunk_number.
    
    Returns the MP4 video file for the requested chunk.
    
    Args:
        session_id: The session identifier (format: agent_id_rule_index_timestamp)
        chunk_number: The chunk number (0-indexed)
        current_user: Current authenticated user (from dependency)
        
    Returns:
        FileResponse with the MP4 video file
        
    Raises:
        HTTPException: If chunk not found or access denied
    """
    try:
        video_path = get_video_chunk_path(session_id, chunk_number)
        
        if not video_path or not Path(video_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video chunk not found",
            )
        
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=f"chunk_{chunk_number}.mp4",
            headers={
                "Content-Disposition": f'attachment; filename="chunk_{chunk_number}.mp4"'
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error retrieving video chunk: session_id=%s, chunk_number=%s: %s",
            session_id,
            chunk_number,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving video chunk",
        )


@router.get("/event-image")
async def get_event_image(
    path: str = Query(..., description="Path to a saved event image under the events directory"),
    current_user: UserResponse = Depends(get_current_user),
) -> FileResponse:
    """
    Serve a saved event image file (JPG/PNG/WEBP) from the events directory.
    The Electron UI uses this to show real event thumbnails/cards.
    """
    base_dir = EVENTS_BASE_DIR.resolve()
    try:
        resolved = Path(path).resolve()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image path")

    # Prevent path traversal: must be under events/
    if base_dir not in resolved.parents and resolved != base_dir:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    if resolved.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported image type")

    return FileResponse(path=str(resolved))


@router.get("/video-chunks/{session_id}/{chunk_number}/metadata")
async def get_video_chunk_metadata(
    session_id: str,
    chunk_number: int,
    current_user: UserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get metadata for a specific video chunk.
    
    Returns the JSON metadata associated with the video chunk, including
    event information, timestamps, and video properties.
    
    Args:
        session_id: The session identifier (format: agent_id_rule_index_timestamp)
        chunk_number: The chunk number (0-indexed)
        current_user: Current authenticated user (from dependency)
        
    Returns:
        Dictionary containing chunk metadata
        
    Raises:
        HTTPException: If metadata not found or access denied
    """
    try:
        metadata_path = get_video_chunk_metadata_path(session_id, chunk_number)
        
        if not metadata_path or not Path(metadata_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video chunk metadata not found",
            )
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error retrieving video chunk metadata: session_id=%s, chunk_number=%s: %s",
            session_id,
            chunk_number,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving video chunk metadata",
        )


# ============================================================================
# Firebase Push Notification Endpoints
# ============================================================================

class DeviceTokenRequest(BaseModel):
    """Request model for device token registration"""
    token: str


@router.post("/register-device", status_code=status.HTTP_200_OK)
async def register_device_token(request: DeviceTokenRequest) -> Dict[str, Any]:
    """
    Register a mobile device FCM token for push notifications.
    
    Stores the token in memory (simple implementation).
    In production, this should be stored in a database per user.
    
    Args:
        request: Device token registration request
        
    Returns:
        Success message with registered token count
    """
    global registered_device_tokens

    token = request.token.strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token cannot be empty",
        )
    if token not in registered_device_tokens:
        registered_device_tokens.append(token)
        logger.info("Device token registered; total tokens: %s", len(registered_device_tokens))
    else:
        logger.debug("Device token already registered")
    return {
        "message": "Device token registered successfully",
        "total_tokens": len(registered_device_tokens),
    }


@router.post("/send-notification", status_code=status.HTTP_200_OK)
async def send_notification(
    request: Request,
    title: Optional[str] = None,
    body: Optional[str] = None,
    image_path: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send push notification to all registered mobile devices.
    
    This endpoint loops through all registered device tokens and sends
    a push notification using Firebase Cloud Messaging.
    
    Args:
        request: FastAPI request object (for building image URLs)
        title: Notification title (default: "Vision Alert")
        body: Notification body (default: "An event has been detected")
        image_path: Optional path to event image (relative to events/ directory)
        token: Optional FCM device token (for single device notification)
    Returns:
        Dictionary with notification results:
        {
            "message": "Notifications sent",
            "total_tokens": int,
            "success_count": int,
            "failure_count": int
        }
    """
    global registered_device_tokens

    if not token and not registered_device_tokens:
        return {
            "message": "No device tokens registered or provided",
            "success_count": 0,
            "failure_count": 0,
        }

    # Default notification content
    notification_title = title or "Vision Alert"
    notification_body = body or "An event has been detected"
    
    # Build image URL if image_path is provided
    image_url: Optional[str] = None
    if image_path:
        try:
            base_url = str(request.base_url).rstrip("/")
            encoded_path = quote(image_path, safe="")
            image_url = f"{base_url}/api/v1/notifications/event-image?path={encoded_path}"
            logger.info("Notification image URL: %s", image_url)
        except Exception as e:
            logger.warning("Failed to build image URL: %s", e)
            image_url = None

    if token:
        result = send_push_notification(
            token=token,
            title=notification_title,
            body=notification_body,
            image_url=image_url,
        )
    else:
        result = send_push_notification_multicast(
            tokens=registered_device_tokens,
            title=notification_title,
            body=notification_body,
            image_url=image_url,
        )

    logger.info(
        "Push notifications sent: %s successful, %s failed",
        result["success_count"],
        result["failure_count"],
    )
    return {
        "message": "Notifications sent",
        "total_tokens": len(registered_device_tokens),
        "success_count": result["success_count"],
        "failure_count": result["failure_count"],
    }

