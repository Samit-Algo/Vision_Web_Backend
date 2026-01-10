# Standard library imports
from typing import Optional

# External package imports
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import Response

# Local application imports
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.camera.get_camera import GetCameraUseCase
from ...infrastructure.streaming import WsFmp4Service
from ...di.container import get_container
from .dependencies import get_current_user
import logging

logger = logging.getLogger(__name__)


router = APIRouter(tags=["streaming"])


async def get_camera_for_user(
    camera_id: str,
    current_user: Optional[UserResponse],
):
    """
    Fetch camera and verify the current user has access.
    
    Args:
        camera_id: ID of the camera
        current_user: Current authenticated user
        
    Raises:
        HTTPException: If camera not found or user doesn't have access
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    container = get_container()
    get_camera_use_case = container.get(GetCameraUseCase)
    
    try:
        return await get_camera_use_case.execute(
            camera_id=camera_id,
            owner_user_id=current_user.id,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found or access denied"
        )


@router.get("/{camera_id}/status")
async def get_stream_status(
    camera_id: str,
    request: Request,
    current_user: UserResponse = Depends(get_current_user),
) -> dict:
    """
    Get status of WebSocket live stream for a camera.
    
    Args:
        camera_id: ID of the camera
        current_user: Current authenticated user
        
    Returns:
        Live stream status information
        
    Raises:
        HTTPException: If camera not found or access denied
    """
    # Verify camera access (and ensure camera exists)
    await get_camera_for_user(camera_id, current_user)
    
    container = get_container()
    ws_service: WsFmp4Service = container.get(WsFmp4Service)

    is_streaming = ws_service.is_streaming(camera_id)
    viewers = ws_service.get_viewer_count(camera_id)
    last_error = ws_service.get_last_error(camera_id)

    ws_scheme = "wss" if request.url.scheme == "https" else "ws"
    host = request.headers.get("host", "localhost")
    ws_url = f"{ws_scheme}://{host}/api/v1/streams/{camera_id}/live/ws"
    
    return {
        "camera_id": camera_id,
        "is_streaming": is_streaming,
        "ws_url": ws_url,
        "viewers": viewers,
        "last_error": last_error,
    }


@router.websocket("/{camera_id}/live/ws")
async def websocket_live_stream(websocket: WebSocket, camera_id: str) -> None:
    """
    WebSocket live stream endpoint.

    Auth:
    - Pass JWT as query param: ?token=...

    Streaming:
    - Server broadcasts fragmented MP4 (fMP4) bytes suitable for MSE on Electron/Chromium.
    - 1 FFmpeg process per camera, shared across viewers.
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Authentication token required")
        return

    # Authenticate from query token (WebSocket clients often can't set Authorization header easily)
    try:
        class MockCredentials:
            credentials = token

        current_user = await get_current_user(MockCredentials())
    except Exception:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return

    # Verify access and fetch camera (for RTSP URL)
    try:
        camera = await get_camera_for_user(camera_id, current_user)
    except HTTPException as e:
        await websocket.close(code=1008, reason=str(e.detail))
        return

    rtsp_url = getattr(camera, "stream_url", None)
    if not rtsp_url:
        await websocket.close(code=1008, reason="Camera does not have a stream URL configured")
        return

    await websocket.accept()

    container = get_container()
    ws_service: WsFmp4Service = container.get(WsFmp4Service)

    await ws_service.add_viewer(camera_id=camera_id, websocket=websocket, rtsp_url=rtsp_url)

    try:
        # Keep the connection open; we don't require client messages.
        while True:
            await websocket.receive()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error for camera %s: %s", camera_id, e, exc_info=True)
    finally:
        await ws_service.remove_viewer(camera_id=camera_id, websocket=websocket)

