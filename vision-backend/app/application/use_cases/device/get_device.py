# Standard library imports
from typing import Optional

# Local application imports
from ....domain.repositories.device_repository import DeviceRepository
from ...dto.device_dto import DeviceResponse


class GetDeviceUseCase:
    """Use case for getting a device by ID"""
    
    def __init__(
        self,
        device_repository: DeviceRepository,
    ) -> None:
        self.device_repository = device_repository
    
    async def execute(
        self,
        device_id: str,
        owner_user_id: str,
    ) -> DeviceResponse:
        """
        Get a device by ID
        
        Args:
            device_id: ID of the device
            owner_user_id: ID of the user (for ownership validation)
            
        Returns:
            DeviceResponse with device information
            
        Raises:
            ValueError: If device not found or doesn't belong to user
        """
        device = await self.device_repository.find_by_id(device_id)
        
        if not device:
            raise ValueError(f"Device {device_id} not found")
        
        if device.owner_user_id != owner_user_id:
            raise ValueError(f"Device {device_id} does not belong to user {owner_user_id}")
        
        return DeviceResponse(
            id=device.id or "",
            owner_user_id=device.owner_user_id,
            name=device.name,
            jetson_backend_url=device.jetson_backend_url,
        )

