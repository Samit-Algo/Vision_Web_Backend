# Standard library imports
import logging
from typing import TYPE_CHECKING

# Local application imports
from ....domain.repositories.device_repository import DeviceRepository
from ....domain.models.device import Device
from ...dto.device_dto import DeviceCreateRequest, DeviceResponse

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CreateDeviceUseCase:
    """Use case for creating a new device"""
    
    def __init__(
        self,
        device_repository: DeviceRepository,
    ) -> None:
        self.device_repository = device_repository
    
    async def execute(
        self,
        request: DeviceCreateRequest,
        owner_user_id: str,
    ) -> DeviceResponse:
        """
        Create a new device
        
        Args:
            request: Device creation request
            owner_user_id: ID of the user creating the device
            
        Returns:
            DeviceResponse with created device information
            
        Raises:
            ValueError: If device_id already exists or validation fails
        """
        # Check if device with this ID already exists
        existing_device = await self.device_repository.find_by_id(request.id)
        if existing_device:
            if existing_device.owner_user_id != owner_user_id:
                raise ValueError(f"Device {request.id} already exists and belongs to another user")
            # If it belongs to the same user, update it
            logger.info(f"Device {request.id} already exists for user {owner_user_id}, updating...")
        
        # Create domain device entity
        new_device = Device(
            id=request.id,
            owner_user_id=owner_user_id,
            name=request.name,
            jetson_backend_url=request.jetson_backend_url,
        )
        
        # Save device
        saved_device = await self.device_repository.save(new_device)
        
        logger.info(f"Successfully saved device {saved_device.id} for user {owner_user_id}")
        
        # Return DTO
        return DeviceResponse(
            id=saved_device.id or "",
            owner_user_id=saved_device.owner_user_id,
            name=saved_device.name,
            jetson_backend_url=saved_device.jetson_backend_url,
        )

