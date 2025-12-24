# Standard library imports
import logging
from typing import Optional, TYPE_CHECKING

# Local application imports
from ....domain.repositories.device_repository import DeviceRepository
from ....domain.models.device import Device
from ...dto.device_dto import DeviceCreateRequest, DeviceResponse
from ....core.config import get_settings
from ....infrastructure.external.device_client import DeviceClient

logger = logging.getLogger(__name__)


class CreateDeviceUseCase:
    """Use case for creating a new device"""
    
    def __init__(
        self,
        device_repository: DeviceRepository,
        jetson_client: Optional["DeviceClient"] = None,
    ) -> None:
        self.device_repository = device_repository
        self.jetson_client = jetson_client
    
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
        
        # Check connection to Jetson backend before saving
        if self.jetson_client:
            # Create DeviceClient with device-specific URL
            jetson_client = DeviceClient(base_url=request.jetson_backend_url)
            
            logger.info(f"Checking connection to Jetson backend at {request.jetson_backend_url}")
            connection_ok = await jetson_client.check_connection()
            
            if not connection_ok:
                raise ValueError(
                    f"Cannot connect to Jetson backend at {request.jetson_backend_url}. "
                    f"Please verify the URL and ensure the Jetson backend is running."
                )
            
            logger.info(f"Successfully connected to Jetson backend at {request.jetson_backend_url}")
        else:
            logger.warning(
                f"JetsonClient not provided. Skipping connection check for device {request.id}"
            )
        
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
        
        # Register device with Jetson backend (send web backend URL)
        if self.jetson_client and saved_device.id:
            # Create DeviceClient with device-specific URL
            jetson_client = DeviceClient(base_url=request.jetson_backend_url)
            
            # Get web backend URL from settings
            settings = get_settings()
            web_backend_url = settings.web_backend_url
            
            logger.info(
                f"Registering device {saved_device.id} with Jetson backend at {request.jetson_backend_url}, "
                f"sending web backend URL: {web_backend_url}"
            )
            
            success = await jetson_client.register_device(
                device_id=saved_device.id,
                web_backend_url=web_backend_url,
                user_id=owner_user_id,
                name=saved_device.name
            )
            
            if success:
                logger.info(
                    f"Successfully registered device {saved_device.id} with Jetson backend. "
                    f"Bidirectional communication established."
                )
            else:
                logger.warning(
                    f"Device {saved_device.id} saved locally but failed to register with Jetson backend. "
                    f"Device will not be able to communicate with Jetson backend until registration succeeds."
                )
        else:
            logger.warning(
                f"JetsonClient not available. Device {saved_device.id} saved locally only. "
                f"Register JetsonClient to enable Jetson backend integration."
            )
        
        # Return DTO
        return DeviceResponse(
            id=saved_device.id or "",
            owner_user_id=saved_device.owner_user_id,
            name=saved_device.name,
            jetson_backend_url=saved_device.jetson_backend_url,
        )

