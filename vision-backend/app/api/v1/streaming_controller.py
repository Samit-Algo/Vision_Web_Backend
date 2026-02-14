"""
Streaming API: live camera stream (WebSocket fMP4), stream status, snapshot, agent overlay WebSocket.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import asyncio
import dataclasses
import logging
import subprocess
from typing import Any, List, Optional

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from fastapi import APIRouter, Depends, HTTPException, Request, status, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.camera.get_camera import GetCameraUseCase
from ...domain.repositories.agent_repository import AgentRepository
from ...di.container import get_container
from ...infrastructure.streaming import WsFmp4Service
from ...processing.helpers import get_shared_store

from .dependencies import get_current_user

# -----------------------------------------------------------------------------
# Logging and router
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Optional: for agent frame-by-frame streaming (JPEG push)
try:
    import numpy as np
    import cv2  # type: ignore
    _FRAME_ENCODE_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    cv2 = None  # type: ignore
    _FRAME_ENCODE_AVAILABLE = False


def _entry_to_jpeg_bytes(entry: Any) -> Optional[bytes]:
    """Encode shared_store frame entry to JPEG bytes. Returns None if invalid or cv2/numpy missing."""
    if not _FRAME_ENCODE_AVAILABLE or not entry or not isinstance(entry, dict):
        return None
    try:
        buf = entry.get("bytes")
        shape = entry.get("shape")
        dtype_str = entry.get("dtype", "uint8")
        if not buf or not shape or len(shape) < 3:
            return None
        arr = np.frombuffer(buf, dtype=np.dtype(dtype_str))
        arr = arr.reshape(tuple(int(x) for x in shape))
        _, jpeg = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes()
    except Exception:
        return None


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


@router.get("/{camera_id}/snapshot.jpg")
async def get_camera_snapshot(
    camera_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> Response:
    """
    Return a single JPEG snapshot for the given camera.

    Used by the desktop chat UI when a rule requires the user to draw a zone.
    """
    camera = await get_camera_for_user(camera_id, current_user)
    rtsp_url = getattr(camera, "stream_url", None)
    if not rtsp_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera does not have a stream URL configured",
        )

    # Use FFmpeg to grab exactly 1 frame as JPEG to stdout.
    # This avoids requiring Pillow/OpenCV in the backend runtime.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-fflags",
        "+discardcorrupt+nobuffer",
        "-flags",
        "low_delay",
        "-use_wallclock_as_timestamps",
        "1",
        "-i",
        rtsp_url,
        "-an",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=6,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Snapshot timed out")
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ffmpeg not found on server")
    except Exception as e:
        logger.error("Snapshot capture failed for camera %s: %s", camera_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Snapshot capture failed")

    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="ignore")[-500:]
        logger.warning("Snapshot ffmpeg failed for camera %s: %s", camera_id, err)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to capture snapshot")

    return Response(
        content=proc.stdout,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )


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

    try:
        class MockCredentials:
            credentials = token
        current_user = await get_current_user(credentials=MockCredentials(), token=token)
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
        # After client disconnect, receive() can raise RuntimeError if called again — so check message type and break.
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "disconnect" not in str(e).lower():
            logger.error("WebSocket error for camera %s: %s", camera_id, e, exc_info=True)
    except Exception as e:
        logger.error("WebSocket error for camera %s: %s", camera_id, e, exc_info=True)
    finally:
        await ws_service.remove_viewer(camera_id=camera_id, websocket=websocket)


@router.websocket("/agents/{agent_id}/overlay/ws")
async def websocket_agent_processed_stream(
    websocket: WebSocket,
    agent_id: str,
) -> None:
    """
    Agent processed-frame stream (fMP4 over WebSocket).

    Streams video with detections already drawn (same fMP4 format as raw live stream).
    Auth: pass JWT as query param ?token=...
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Authentication token required")
        return

    try:
        class MockCredentials:
            credentials = token
        current_user = await get_current_user(credentials=MockCredentials(), token=token)
    except Exception:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return

    container = get_container()
    agent_repo: AgentRepository = container.get(AgentRepository)
    agent = await agent_repo.find_by_id(agent_id)
    if not agent or not agent.owner_user_id or str(agent.owner_user_id) != str(current_user.id):
        await websocket.close(code=1008, reason="Agent not found or access denied")
        return

    shared_store = get_shared_store()
    if shared_store is None:
        await websocket.close(code=1011, reason="Shared store not initialized")
        return

    await websocket.accept()

    processed_service: ProcessedFrameStreamService = container.get(
        ProcessedFrameStreamService
    )
    await processed_service.add_viewer(
        agent_id=agent_id,
        websocket=websocket,
        shared_store=shared_store,
    )

    try:
        # After client disconnect, receive() can raise RuntimeError if called again — check message type and break.
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "disconnect" not in str(e).lower():
            logger.error(
                "WebSocket error for agent processed stream %s: %s",
                agent_id,
                e,
                exc_info=True,
            )
    except Exception as e:
        logger.error(
            "WebSocket error for agent processed stream %s: %s",
            agent_id,
            e,
            exc_info=True,
        )
    finally:
        await processed_service.remove_viewer(agent_id=agent_id, websocket=websocket)


@router.websocket("/agents/{agent_id}/overlay/frames/ws")
async def websocket_agent_processed_frames(
    websocket: WebSocket,
    agent_id: str,
) -> None:
    """
    Agent processed frames as JPEG over WebSocket (one message per frame).

    Sends frames directly to the UI so the stream does not depend on fMP4/chunk pipeline
    and does not break when encoding or buffering stalls. Auth: ?token=...
    """
    if not _FRAME_ENCODE_AVAILABLE:
        await websocket.close(code=1011, reason="Frame encoding (numpy/cv2) not available")
        return

    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Authentication token required")
        return

    try:
        class MockCredentials:
            credentials = token
        current_user = await get_current_user(MockCredentials())
    except Exception:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return

    container = get_container()
    agent_repo: AgentRepository = container.get(AgentRepository)
    agent = await agent_repo.find_by_id(agent_id)
    if not agent or not agent.owner_user_id or str(agent.owner_user_id) != str(current_user.id):
        await websocket.close(code=1008, reason="Agent not found or access denied")
        return

    shared_store = get_shared_store()
    if shared_store is None:
        await websocket.close(code=1011, reason="Shared store not initialized")
        return

    await websocket.accept()

    _frame_push_done = asyncio.Event()
    _frame_push_task: Optional[asyncio.Task] = None

    async def frame_push_loop() -> None:
        max_fps = 15.0
        interval = 1.0 / max_fps
        last_jpeg: Optional[bytes] = None
        try:
            while not _frame_push_done.is_set():
                entry = None
                try:
                    entry = shared_store.get(agent_id)
                except Exception:
                    pass
                jpeg = _entry_to_jpeg_bytes(entry) if entry else None
                if jpeg:
                    last_jpeg = jpeg
                if last_jpeg:
                    try:
                        await asyncio.wait_for(
                            websocket.send_bytes(last_jpeg),
                            timeout=2.0,
                        )
                    except asyncio.TimeoutError:
                        pass
                    except Exception:
                        break
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        finally:
            _frame_push_done.set()

    try:
        _frame_push_task = asyncio.create_task(frame_push_loop())
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "disconnect" not in str(e).lower():
            logger.error("WebSocket error for agent frames %s: %s", agent_id, e, exc_info=True)
    except Exception as e:
        logger.error("WebSocket error for agent frames %s: %s", agent_id, e, exc_info=True)
    finally:
        _frame_push_done.set()
        if _frame_push_task and not _frame_push_task.done():
            _frame_push_task.cancel()
            try:
                await _frame_push_task
            except asyncio.CancelledError:
                pass

