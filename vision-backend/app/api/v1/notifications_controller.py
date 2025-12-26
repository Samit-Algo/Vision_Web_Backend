"""Notifications API endpoints for real-time event notifications via WebSocket"""

import logging
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException, status, Depends, Response
from fastapi.responses import FileResponse

from ...core.security import decode_jwt_token
from ...infrastructure.notifications import WebSocketManager
from ...utils.event_storage import (
    get_video_chunk_path,
    get_video_chunk_metadata_path,
    list_video_chunks_for_session
)
from .dependencies import get_current_user
from ...application.dto.user_dto import UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["notifications"])

# Global WebSocket manager (set from main.py)
_global_websocket_manager: Optional[WebSocketManager] = None


def set_websocket_manager(manager: WebSocketManager) -> None:
    """Set the global WebSocket manager instance"""
    global _global_websocket_manager
    _global_websocket_manager = manager


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance"""
    if _global_websocket_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WebSocket manager not available"
        )
    return _global_websocket_manager


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
        logger.warning(f"Invalid token for WebSocket connection: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}"
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
    logger.info(f"WebSocket connection accepted for user {user_id}")
    
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
                    logger.debug(f"Received message from user {user_id}: {message}")
                    
            except WebSocketDisconnect:
                # Client disconnected
                logger.info(f"WebSocket disconnected for user {user_id}")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message for user {user_id}: {e}", exc_info=True)
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket connection for user {user_id}: {e}", exc_info=True)
    finally:
        # Clean up: remove connection from manager
        try:
            await manager.remove_connection(user_id, websocket)
            logger.info(f"WebSocket connection cleaned up for user {user_id}")
        except Exception as e:
            logger.error(f"Error cleaning up WebSocket connection for user {user_id}: {e}", exc_info=True)


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
                detail=f"No video chunks found for session_id: {session_id}"
            )
        
        return {
            "session_id": session_id,
            "chunks": chunks
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing video chunks for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing video chunks: {str(e)}"
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
                detail=f"Video chunk not found: session_id={session_id}, chunk_number={chunk_number}"
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
            f"Error retrieving video chunk: session_id={session_id}, chunk_number={chunk_number}, error={e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving video chunk: {str(e)}"
        )


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
                detail=f"Video chunk metadata not found: session_id={session_id}, chunk_number={chunk_number}"
            )
        
        import json
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error retrieving video chunk metadata: session_id={session_id}, chunk_number={chunk_number}, error={e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving video chunk metadata: {str(e)}"
        )

