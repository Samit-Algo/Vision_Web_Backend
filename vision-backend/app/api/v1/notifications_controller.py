"""Notifications API endpoints for real-time event notifications via WebSocket"""

import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException, status

from ...core.security import decode_jwt_token
from ...infrastructure.notifications import WebSocketManager

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

