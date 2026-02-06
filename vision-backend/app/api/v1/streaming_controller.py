# Standard library imports
import dataclasses
from typing import Any, List, Optional
import subprocess

# External package imports
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import Response

# Local application imports
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.camera.get_camera import GetCameraUseCase
from ...domain.repositories.agent_repository import AgentRepository
from ...infrastructure.streaming import WsFmp4Service
from ...di.container import get_container
from ...processing.helpers import get_shared_store
from .dependencies import get_current_user
import logging
import asyncio

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


@router.websocket("/agents/{agent_id}/overlay/ws")
async def websocket_agent_overlay(websocket: WebSocket, agent_id: str) -> None:
    """
    Agent overlay websocket endpoint (Option A):
    - Keep camera video stream unchanged (fMP4)
    - Stream only detection metadata for the selected agent

    Client draws overlays (boxes/labels) over the camera video element.
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Authentication token required")
        return

    # Authenticate from query token
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

    last_frame_index = None
    try:
        while True:
            entry = None
            try:
                entry = shared_store.get(agent_id, None)
            except Exception:
                entry = None

            if not isinstance(entry, dict):
                # No agent output yet (agent not running / no frames) - heartbeat
                await websocket.send_json({"agent_id": agent_id, "status": "no_data"})
                await asyncio.sleep(0.25)
                continue

            frame_index = entry.get("frame_index")
            if frame_index is None or frame_index == last_frame_index:
                await asyncio.sleep(0.03)
                continue
            last_frame_index = frame_index

            shape = entry.get("shape") or ()
            try:
                height = int(shape[0])
                width = int(shape[1])
            except Exception:
                height = None
                width = None

            det = entry.get("detections") or {}
            boxes = det.get("boxes") or []
            classes = det.get("classes") or []
            scores = det.get("scores") or []
            keypoints = det.get("keypoints") or []  # For fall_detection/pose (skeleton overlay)
            
            # Get scenario overlays (e.g., loom ROI boxes with state labels)
            # Ensure JSON-serializable (pipeline stores dicts; convert any dataclass leftovers)
            raw_overlays = entry.get("scenario_overlays") or []
            scenario_overlays: List[Any] = []
            for item in raw_overlays:
                if dataclasses.is_dataclass(item) and not isinstance(item, type):
                    scenario_overlays.append(dataclasses.asdict(item))
                elif isinstance(item, dict):
                    scenario_overlays.append(item)
                else:
                    scenario_overlays.append(item)

            # Collect zones from agent configuration and rules
            zones = []
            
            # Add agent-level zone if exists
            if agent and hasattr(agent, "zone") and agent.zone:
                zones.append(agent.zone)
            
            # Add zones from rules (including line zones for class_count/box_count)
            rules = entry.get("rules", [])
            if rules:
                for rule in rules:
                    rule_zone = rule.get("zone")
                    if rule_zone:
                        zones.append(rule_zone)
            
            # Also check for line_zone in entry (from scenario)
            line_zone = entry.get("line_zone")
            if line_zone and line_zone not in zones:
                zones.append(line_zone)
            
            # Get zone violation status from entry
            zone_violated = entry.get("zone_violated", False)
            
            # Get fire detection status (for red bounding boxes on overlay)
            fire_detected = entry.get("fire_detected", False)
            
            # Get line crossing/touch status
            line_crossed = entry.get("line_crossed", False)
            line_crossed_indices = entry.get("line_crossed_indices", [])
            track_info = entry.get("track_info", [])  # Track information with center points and touch status
            in_zone_indices = entry.get("in_zone_indices", [])  # Restricted zone: only these detection indices get red box
            sleep_confirmed_indices = entry.get("sleep_confirmed_indices", [])  # Sleep: same person box, red when VLM confirmed
            wall_climb_red_indices = entry.get("wall_climb_red_indices", [])  # Wall climb: fully above (stays red)
            wall_climb_orange_indices = entry.get("wall_climb_orange_indices", [])  # Wall climb: climbing (orange)

            # Build detection colors based on touch status
            # Yellow for boxes touching the line, default color for others
            detection_colors = []
            for idx, track_item in enumerate(track_info):
                if track_item.get("touching_line", False):
                    detection_colors.append("yellow")  # Yellow when touching line
                else:
                    detection_colors.append("green")  # Default green for normal boxes
            
            payload = {
                "type": "agent_overlay",
                "agent_id": agent_id,
                "camera_id": getattr(agent, "camera_id", None),
                "frame_index": frame_index,
                "ts_monotonic": entry.get("ts_monotonic"),  # Timestamp for staleness check
                "width": width,
                "height": height,
                "detections": {
                    "boxes": boxes,
                    "classes": classes,
                    "scores": scores,
                    "colors": detection_colors,  # Box colors: yellow when touching, green otherwise
                    "keypoints": keypoints,  # For fall_detection/pose UI (skeleton overlay)
                },
                "scenario_overlays": scenario_overlays,
                "zones": zones,
                "zone": zones[0] if zones else None,
                "zone_violated": zone_violated,
                "fire_detected": fire_detected,
                "line_crossed": line_crossed,
                "line_crossed_indices": line_crossed_indices,
                "track_info": track_info,
                "in_zone_indices": in_zone_indices,
                "sleep_confirmed_indices": sleep_confirmed_indices,
                "wall_climb_red_indices": wall_climb_red_indices,
                "wall_climb_orange_indices": wall_climb_orange_indices,
            }
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        return
    except Exception as e:
        logger.error("WebSocket overlay error for agent %s: %s", agent_id, e, exc_info=True)
        try:
            await websocket.close(code=1011, reason="Overlay stream error")
        except Exception:
            pass

