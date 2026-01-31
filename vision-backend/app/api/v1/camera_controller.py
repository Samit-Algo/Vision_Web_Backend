# Standard library imports
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np

# External package imports
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

# Local application imports
from ...application.dto.camera_dto import CameraCreateRequest, CameraResponse, WebRTCConfig
from ...application.dto.agent_dto import AgentResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.camera.create_camera import CreateCameraUseCase
from ...application.use_cases.camera.list_cameras import ListCamerasUseCase
from ...application.use_cases.camera.get_camera import GetCameraUseCase
from ...application.use_cases.agent.list_agents_by_camera import ListAgentsByCameraUseCase
from ...infrastructure.external.camera_client import CameraClient
from ...di.container import get_container
from ...processing.helpers import get_shared_store
<<<<<<< HEAD
=======
from ...processing.data_input.hub_source import reconstruct_frame
>>>>>>> 00ff0767b9ab495af597be3941b3bbcb8c46cc96
from ...utils.event_notifier import encode_frame_to_base64
from .dependencies import get_current_user

# Try to import cv2 for saving frames
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def _reconstruct_frame(entry: Dict[str, Any]) -> Optional[np.ndarray]:
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


router = APIRouter(tags=["cameras"])


@router.post("/create", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    request: CameraCreateRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> CameraResponse:
    """
    Create a new camera
    
    Args:
        request: Camera creation request
        current_user: Current authenticated user (from dependency)
        
    Returns:
        CameraResponse with created camera information
    """
    container = get_container()
    create_camera_use_case = container.get(CreateCameraUseCase)
    
    try:
        camera = await create_camera_use_case.execute(
            request=request,
            owner_user_id=current_user.id,
        )
        return camera
    except ValueError as exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exception)
        )


@router.get("/list", response_model=List[CameraResponse])
async def list_cameras(
    current_user: UserResponse = Depends(get_current_user),
) -> List[CameraResponse]:
    """
    List all cameras for the current user
    
    Args:
        current_user: Current authenticated user (from dependency)
        
    Returns:
        List of CameraResponse objects
    """
    container = get_container()
    list_cameras_use_case = container.get(ListCamerasUseCase)
    
    cameras = await list_cameras_use_case.execute(owner_user_id=current_user.id)
    return cameras


@router.get("/get/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> CameraResponse:
    """
    Get a camera by ID
    
    Args:
        camera_id: ID of the camera
        current_user: Current authenticated user (from dependency)
        
    Returns:
        CameraResponse with camera information
    """
    container = get_container()
    get_camera_use_case = container.get(GetCameraUseCase)
    
    try:
        camera = await get_camera_use_case.execute(
            camera_id=camera_id,
            owner_user_id=current_user.id,
        )
        return camera
    except ValueError as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exception)
        )


@router.get("/{camera_id}/agents", response_model=List[AgentResponse])
async def list_agents_by_camera(
    camera_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> List[AgentResponse]:
    """
    List all agents for a specific camera
    
    Args:
        camera_id: ID of the camera
        current_user: Current authenticated user (from dependency)
        
    Returns:
        List of AgentResponse objects for the camera
    """
    container = get_container()
    list_agents_use_case = container.get(ListAgentsByCameraUseCase)
    
    try:
        agents = await list_agents_use_case.execute(
            camera_id=camera_id,
            user_id=current_user.id,
        )
        return agents
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing agents for camera: {str(e)}"
        )


@router.get("/webrtc-config", response_model=WebRTCConfig)
async def get_webrtc_config(
    current_user: UserResponse = Depends(get_current_user),
) -> WebRTCConfig:
    """
    Get WebRTC configuration for streaming cameras.
    
    This endpoint retrieves the signaling server URL and ICE servers
    needed for the frontend to establish WebRTC connections with the
    Jetson backend for live camera streaming.
    
    Args:
        current_user: Current authenticated user (from dependency)
        
    Returns:
        WebRTCConfig with signaling URL, viewer ID, and ICE servers
        
    Raises:
        HTTPException: If Jetson backend is unavailable or user has no cameras
    """
    container = get_container()
    camera_client = container.get(CameraClient)
    
    # Get WebRTC configuration from Jetson backend
    config = await camera_client.get_webrtc_config(current_user.id)
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Jetson backend unavailable or no cameras registered for this user"
        )
    
    # Transform Jetson backend response to our DTO format
    return WebRTCConfig(
        signaling_url=config.get("signaling_url", ""),
        viewer_id=f"viewer:{current_user.id}",
        ice_servers=config.get("ice_servers", [])
    )


async def get_camera_for_user(
    camera_id: str,
    current_user: UserResponse,
):
    """
    Fetch camera and verify the current user has access.
    
    Args:
        camera_id: ID of the camera
        current_user: Current authenticated user
        
    Raises:
        HTTPException: If camera not found or user doesn't have access
    """
    get_camera_use_case = get_container().get(GetCameraUseCase)
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


@router.get("/{camera_id}/snapshot")
async def get_camera_snapshot(
    camera_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> JSONResponse:
    """
    Get a snapshot frame from a camera for zone drawing.
    
    Returns the latest frame from the camera as a base64-encoded JPEG image.
    This endpoint is used when the user needs to draw a zone on the camera view.
    
    Args:
        camera_id: ID of the camera
        current_user: Current authenticated user (from dependency)
        
    Returns:
        JSONResponse with base64-encoded frame and metadata
        
    Raises:
        HTTPException: If camera not found, access denied, or frame unavailable
    """
    print(f"[SNAPSHOT] 📸 Snapshot requested for camera_id={camera_id}, user_id={current_user.id}")
    
    # Verify camera access and get camera object (we'll need it for RTSP URL if fallback is needed)
    camera = await get_camera_for_user(camera_id, current_user)
    print(f"[SNAPSHOT] ✅ Camera access verified for camera_id={camera_id}")
    
    # Get shared store and retrieve frame
    shared_store = get_shared_store()
    if shared_store is None:
        print(f"[SNAPSHOT] ❌ Shared store is None for camera_id={camera_id}")
        # Don't fail yet - try RTSP fallback
    
    frame = None
    
    try:
        # Try to get frame from shared_store first (if available)
        if shared_store is not None:
            print(f"[SNAPSHOT] 📦 Shared store available, checking for camera_id={camera_id}")
            print(f"[SNAPSHOT] 📋 Available keys in shared_store: {list(shared_store.keys()) if hasattr(shared_store, 'keys') else 'N/A'}")
            
            entry = shared_store.get(camera_id, None)
            print(f"[SNAPSHOT] 🔍 Retrieved entry for camera_id={camera_id}: type={type(entry)}, is_dict={isinstance(entry, dict)}")
            
            # Try to get frame from shared_store first
            if isinstance(entry, dict):
                print(f"[SNAPSHOT] 📊 Entry keys: {list(entry.keys())}")
                print(f"[SNAPSHOT] 📊 Entry has 'bytes': {'bytes' in entry}, has 'shape': {'shape' in entry}, has 'dtype': {'dtype' in entry}")
                
                # Reconstruct frame from shared store
                frame = _reconstruct_frame(entry)
                if frame is not None:
                    print(f"[SNAPSHOT] ✅ Frame reconstructed from shared_store: shape={frame.shape}, dtype={frame.dtype}")
        
        # Fallback: Capture frame directly from RTSP stream if shared_store doesn't have it
        if frame is None:
            print(f"[SNAPSHOT] ⚠️  No frame in shared_store, attempting to capture directly from RTSP stream...")
            
            # Get RTSP URL from camera object (already fetched above)
            rtsp_url = getattr(camera, "stream_url", None) or getattr(camera, "rtsp_url", None)
            
            if not rtsp_url:
                print(f"[SNAPSHOT] ❌ Camera has no RTSP URL configured")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Camera does not have a stream URL configured"
                )
            
            # Try to capture a frame using PyAV (same method as CameraPublisher)
            try:
                import av
                print(f"[SNAPSHOT] 🎥 Connecting to RTSP stream: {rtsp_url}")
                
                rtsp_container = av.open(
                    rtsp_url,
                    format="rtsp",
                    options={
                        "rtsp_transport": "tcp",
                        "max_delay": "0",
                    },
                )
                
                video_stream = rtsp_container.streams.video[0]
                
                # Decode first frame
                frame_decoded = None
                for frame_packet in rtsp_container.decode(video_stream):
                    frame_decoded = frame_packet
                    break  # Only need first frame
                
                rtsp_container.close()
                
                if frame_decoded:
                    # Convert to BGR numpy array (same format as CameraPublisher)
                    frame_bgr = frame_decoded.to_ndarray(format="bgr24")
                    frame = frame_bgr
                    print(f"[SNAPSHOT] ✅ Frame captured from RTSP: shape={frame.shape}, dtype={frame.dtype}")
                else:
                    print(f"[SNAPSHOT] ❌ Failed to decode frame from RTSP stream")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Failed to capture frame from camera stream. The camera may be offline or unreachable."
                    )
                    
            except ImportError:
                print(f"[SNAPSHOT] ❌ PyAV not available for RTSP capture")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Frame capture service not available. Please ensure the camera is streaming."
                )
            except Exception as rtsp_error:
                print(f"[SNAPSHOT] ❌ Error capturing from RTSP: {str(rtsp_error)}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Failed to capture frame from camera: {str(rtsp_error)}"
                )
        
        if frame is None:
            print(f"[SNAPSHOT] ❌ Failed to get frame from any source")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No frame available for this camera. Make sure the camera is streaming."
            )
        
        # Encode frame to base64
        frame_base64 = encode_frame_to_base64(frame)
        if not frame_base64:
            print(f"[SNAPSHOT] ❌ Failed to encode frame to base64 for camera_id={camera_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to encode frame"
            )
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        print(f"[SNAPSHOT] ✅ Frame encoded successfully: width={width}, height={height}, base64_length={len(frame_base64)}")
        
        # Save frame to debug folder for debugging (store the frame being sent to frontend)
        try:
            import os
            # Get the project root (vision-backend directory)
            # This file is at: vision-backend/app/api/v1/camera_controller.py
            # So we go up 3 levels: .. -> app -> .. -> vision-backend
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent  # vision-backend directory
            debug_frames_dir = project_root / "debug_frames" / camera_id
            debug_frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            frame_filename = f"snapshot_{timestamp}.jpg"
            frame_path = debug_frames_dir / frame_filename
            
            if CV2_AVAILABLE:
                # Save frame using cv2
                cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                print(f"[SNAPSHOT] 💾 Frame saved to: {frame_path}")
                print(f"[SNAPSHOT] 💾 Frame file size: {frame_path.stat().st_size} bytes")
            else:
                # Fallback: save base64 data directly
                import base64
                frame_bytes = base64.b64decode(frame_base64)
                frame_path.write_bytes(frame_bytes)
                print(f"[SNAPSHOT] 💾 Frame saved (base64) to: {frame_path}")
                print(f"[SNAPSHOT] 💾 Frame file size: {len(frame_bytes)} bytes")
        except Exception as save_error:
            # Don't fail the request if saving fails
            print(f"[SNAPSHOT] ⚠️  Failed to save debug frame: {str(save_error)}")
        
        # Log response structure for debugging
        response_data = {
            "camera_id": camera_id,
            "frame_base64": frame_base64,
            "width": width,
            "height": height,
            "format": "jpeg"
        }
        print(f"[SNAPSHOT] 📤 Sending response to frontend:")
        print(f"[SNAPSHOT]    - camera_id: {camera_id}")
        print(f"[SNAPSHOT]    - width: {width}, height: {height}")
        print(f"[SNAPSHOT]    - format: jpeg")
        print(f"[SNAPSHOT]    - base64_length: {len(frame_base64)} characters")
        print(f"[SNAPSHOT]    - base64_preview: {frame_base64[:80]}...")
        print(f"[SNAPSHOT] ✅ Response ready to send to frontend")
        
        return JSONResponse(response_data)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SNAPSHOT] ❌ Exception in snapshot endpoint for camera_id={camera_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving camera snapshot: {str(e)}"
        )

