"""
Camera API: create, list, get, list agents, and snapshot (frame for zone drawing).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...application.dto.agent_dto import AgentResponse
from ...application.dto.camera_dto import CameraCreateRequest, CameraResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.camera.create_camera import CreateCameraUseCase
from ...application.use_cases.camera.get_camera import GetCameraUseCase
from ...application.use_cases.camera.list_cameras import ListCamerasUseCase
from ...application.use_cases.agent.list_agents_by_camera import ListAgentsByCameraUseCase
from ...di.container import get_container
from ...processing.helpers import get_shared_store
from ...utils.event_notifier import encode_frame_to_base64

from .dependencies import get_current_user

# -----------------------------------------------------------------------------
# Optional cv2 for saving debug frames
# -----------------------------------------------------------------------------
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
import logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
router = APIRouter(tags=["cameras"])


def reconstruct_frame_from_store_entry(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """Rebuild a numpy array from a shared_store frame entry. Returns None if invalid."""
    try:
        if not entry or "bytes" not in entry or "shape" not in entry or "dtype" not in entry:
            return None
        buffer_bytes = entry["bytes"]
        shape: Tuple[int, ...] = tuple(entry["shape"])
        dtype = np.dtype(entry["dtype"])
        flat_array = np.frombuffer(buffer_bytes, dtype=dtype)
        expected_size = int(shape[0]) * int(shape[1]) * int(shape[2])
        if flat_array.size != expected_size:
            return None
        return flat_array.reshape(shape)
    except Exception:
        return None


async def get_camera_for_user(
    camera_id: str,
    current_user: UserResponse,
):
    """Fetch camera and verify the current user has access. Raises HTTPException if not."""
    get_camera_use_case = get_container().get(GetCameraUseCase)
    try:
        return await get_camera_use_case.execute(
            camera_id=camera_id,
            owner_user_id=current_user.id,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found or access denied",
        )


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post("/create", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    request: CameraCreateRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> CameraResponse:
    """Create a new camera for the current user."""
    container = get_container()
    use_case = container.get(CreateCameraUseCase)
    try:
        camera = await use_case.execute(
            request=request,
            owner_user_id=current_user.id,
        )
        return camera
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/list", response_model=List[CameraResponse])
async def list_cameras(
    current_user: UserResponse = Depends(get_current_user),
) -> List[CameraResponse]:
    """List all cameras for the current user."""
    container = get_container()
    use_case = container.get(ListCamerasUseCase)
    return await use_case.execute(owner_user_id=current_user.id)


@router.get("/get/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> CameraResponse:
    """Get a camera by ID (user must own it)."""
    container = get_container()
    use_case = container.get(GetCameraUseCase)
    try:
        return await use_case.execute(
            camera_id=camera_id,
            owner_user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get("/{camera_id}/agents", response_model=List[AgentResponse])
async def list_agents_by_camera(
    camera_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> List[AgentResponse]:
    """List all agents for a camera (user must have access)."""
    container = get_container()
    use_case = container.get(ListAgentsByCameraUseCase)
    try:
        return await use_case.execute(
            camera_id=camera_id,
            user_id=current_user.id,
        )
    except Exception as e:
        logger.exception("Error listing agents for camera %s", camera_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error listing agents for camera",
        )


@router.get("/{camera_id}/snapshot")
async def get_camera_snapshot(
    camera_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> JSONResponse:
    """
    Get latest frame from camera as base64 JPEG (for zone drawing in UI).

    Tries shared store first; falls back to capturing one frame from RTSP if needed.
    """
    camera = await get_camera_for_user(camera_id, current_user)
    shared_store = get_shared_store()
    frame = None

    if shared_store is not None:
        entry = shared_store.get(camera_id)
        if isinstance(entry, dict):
            frame = reconstruct_frame_from_store_entry(entry)

    if frame is None:
        rtsp_url = getattr(camera, "stream_url", None) or getattr(camera, "rtsp_url", None)
        if not rtsp_url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera does not have a stream URL configured",
            )
        try:
            import av
            container_av = av.open(
                rtsp_url,
                format="rtsp",
                options={"rtsp_transport": "tcp", "max_delay": "0"},
            )
            video_stream = container_av.streams.video[0]
            frame_decoded = None
            for packet in container_av.decode(video_stream):
                frame_decoded = packet
                break
            container_av.close()
            if frame_decoded is not None:
                frame = frame_decoded.to_ndarray(format="bgr24")
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Failed to capture frame from camera stream.",
                )
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Frame capture service not available.",
            )
        except Exception as e:
            logger.warning("RTSP capture failed for camera %s: %s", camera_id, e)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Failed to capture frame from camera.",
            )

    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No frame available for this camera.",
        )

    frame_base64 = encode_frame_to_base64(frame)
    if not frame_base64:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode frame",
        )

    height, width = frame.shape[:2]

    # Optional: save debug frame to disk (non-fatal if it fails)
    try:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        debug_dir = project_root / "debug_frames" / camera_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_path = debug_dir / f"snapshot_{timestamp}.jpg"
        if CV2_AVAILABLE:
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            frame_path.write_bytes(base64.b64decode(frame_base64))
        logger.debug("Snapshot saved to %s", frame_path)
    except Exception as e:
        logger.debug("Could not save debug snapshot: %s", e)

    return JSONResponse({
        "camera_id": camera_id,
        "frame_base64": frame_base64,
        "width": width,
        "height": height,
        "format": "jpeg",
    })
