# Standard library imports
from typing import List

# Local application imports
from ....domain.repositories.device_repository import DeviceRepository
from ...dto.device_dto import DeviceResponse


class ListDevicesUseCase:
    """Use case for listing all devices for a user"""
    
    def __init__(
        self,
        device_repository: DeviceRepository,
    ) -> None:
        self.device_repository = device_repository
    
    async def execute(
        self,
        owner_user_id: str,
    ) -> List[DeviceResponse]:
        """
        List all devices for a user
        
        Args:
            owner_user_id: ID of the user
            
        Returns:
            List of DeviceResponse objects
        """
        devices = await self.device_repository.find_by_owner(owner_user_id)
        
        return [
            DeviceResponse(
                id=device.id or "",
                owner_user_id=device.owner_user_id,
                name=device.name,
                jetson_backend_url=device.jetson_backend_url,
            )
            for device in devices
        ]

