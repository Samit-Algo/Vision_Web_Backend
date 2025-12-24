# Standard library imports
from typing import List

# External package imports
from fastapi import APIRouter, Depends, HTTPException, status

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
from .dependencies import get_current_user


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

