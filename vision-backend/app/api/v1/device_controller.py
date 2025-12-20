# Standard library imports
from typing import List

# External package imports
from fastapi import APIRouter, Depends, HTTPException, status

# Local application imports
from ...application.dto.device_dto import DeviceCreateRequest, DeviceResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.device.create_device import CreateDeviceUseCase
from ...application.use_cases.device.list_devices import ListDevicesUseCase
from ...application.use_cases.device.get_device import GetDeviceUseCase
from ...di.container import get_container
from .dependencies import get_current_user


router = APIRouter(tags=["devices"])


@router.post("/create", response_model=DeviceResponse, status_code=status.HTTP_201_CREATED)
async def create_device(
    request: DeviceCreateRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> DeviceResponse:
    """
    Create a new device (register a Jetson device)
    
    Args:
        request: Device creation request
        current_user: Current authenticated user (from dependency)
        
    Returns:
        DeviceResponse with created device information
    """
    container = get_container()
    create_device_use_case = container.get(CreateDeviceUseCase)
    
    try:
        device = await create_device_use_case.execute(
            request=request,
            owner_user_id=current_user.id,
        )
        return device
    except ValueError as exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exception)
        )


@router.get("/list", response_model=List[DeviceResponse])
async def list_devices(
    current_user: UserResponse = Depends(get_current_user),
) -> List[DeviceResponse]:
    """
    List all devices for the current user
    
    Args:
        current_user: Current authenticated user (from dependency)
        
    Returns:
        List of DeviceResponse objects
    """
    container = get_container()
    list_devices_use_case = container.get(ListDevicesUseCase)
    
    devices = await list_devices_use_case.execute(owner_user_id=current_user.id)
    return devices


@router.get("/get/{device_id}", response_model=DeviceResponse)
async def get_device(
    device_id: str,
    current_user: UserResponse = Depends(get_current_user),
) -> DeviceResponse:
    """
    Get a device by ID
    
    Args:
        device_id: ID of the device
        current_user: Current authenticated user (from dependency)
        
    Returns:
        DeviceResponse with device information
    """
    container = get_container()
    get_device_use_case = container.get(GetDeviceUseCase)
    
    try:
        device = await get_device_use_case.execute(
            device_id=device_id,
            owner_user_id=current_user.id,
        )
        return device
    except ValueError as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exception)
        )

