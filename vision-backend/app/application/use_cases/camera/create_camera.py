# Standard library imports
import secrets
import logging
from typing import Optional, TYPE_CHECKING

# External package imports
import httpx

# Local application imports
from ....domain.repositories.camera_repository import CameraRepository
from ....domain.repositories.user_repository import UserRepository
from ....domain.repositories.device_repository import DeviceRepository
from ....domain.models.camera import Camera
from ...dto.camera_dto import CameraCreateRequest, CameraResponse, WebRTCConfig

if TYPE_CHECKING:
    from ....infrastructure.external.jetson_client import JetsonClient

logger = logging.getLogger(__name__)


class CreateCameraUseCase:
    """Use case for creating a new camera"""
    
    def __init__(
        self,
        camera_repository: CameraRepository,
        user_repository: UserRepository,
        device_repository: DeviceRepository,
        jetson_client: Optional["JetsonClient"] = None,
    ) -> None:
        self.camera_repository = camera_repository
        self.user_repository = user_repository
        self.device_repository = device_repository
        self.jetson_client = jetson_client
    
    def _generate_camera_id(self) -> str:
        """
        Generate a unique camera ID
        
        Returns:
            Unique camera ID string in format CAM-XXXXXXXX
        """
        return f"CAM-{secrets.token_hex(6).upper()}"
    
    async def execute(
        self,
        request: CameraCreateRequest,
        owner_user_id: str,
    ) -> CameraResponse:
        """
        Create a new camera
        
        Args:
            request: Camera creation request
            owner_user_id: ID of the user creating the camera
            
        Returns:
            CameraResponse with created camera information
            
        Raises:
            ValueError: If device_id is invalid or doesn't belong to user
        """
        # Get device backend URL if device_id is provided
        jetson_backend_url = None
        if request.device_id:
            device = await self.device_repository.find_by_id(request.device_id)
            if not device:
                raise ValueError(f"Device {request.device_id} not found")
            if device.owner_user_id != owner_user_id:
                raise ValueError(f"Device {request.device_id} does not belong to user {owner_user_id}")
            jetson_backend_url = device.jetson_backend_url
            logger.info(
                f"Found device {request.device_id} with Jetson backend at {jetson_backend_url}"
            )
        
        # Generate camera ID
        camera_id = self._generate_camera_id()
        
        # Create domain camera entity
        new_camera = Camera(
            id=camera_id,
            owner_user_id=owner_user_id,
            name=request.name,
            stream_url=request.stream_url,
            device_id=request.device_id,
        )
        
        # Save camera
        saved_camera = await self.camera_repository.save(new_camera)
        
        # Register camera with Jetson backend and get WebRTC config
        if saved_camera.id:
            # Create JetsonClient with device-specific URL if available
            jetson_client = self.jetson_client
            if jetson_backend_url:
                if self.jetson_client:
                    # Create a new client instance with device-specific URL
                    from ....infrastructure.external.jetson_client import JetsonClient
                    jetson_client = JetsonClient(base_url=jetson_backend_url)
                    logger.info(
                        f"Using device-specific Jetson backend URL: {jetson_backend_url} for camera {saved_camera.id}"
                    )
                else:
                    logger.warning(
                        f"Device {request.device_id} has Jetson backend URL {jetson_backend_url} "
                        f"but JetsonClient not provided to use case. Camera will be saved locally only."
                    )
            
            if jetson_client:
                try:
                    # Register camera with Jetson backend
                    logger.info(
                        f"Registering camera {saved_camera.id} with Jetson backend at {jetson_client.base_url}"
                    )
                    success = await jetson_client.register_camera(
                        camera_id=saved_camera.id,
                        owner_user_id=saved_camera.owner_user_id,
                        name=saved_camera.name,
                        stream_url=saved_camera.stream_url,
                        device_id=saved_camera.device_id
                    )
                    
                    if success:
                        # Get WebRTC config for this specific camera
                        logger.info(
                            f"Fetching WebRTC config for camera {saved_camera.id} from Jetson backend"
                        )
                        config_dict = await jetson_client.get_webrtc_config_for_camera(
                            user_id=saved_camera.owner_user_id,
                            camera_id=saved_camera.id
                        )
                        
                        if config_dict:
                            # Store WebRTC config in camera
                            saved_camera.webrtc_config = config_dict
                            saved_camera = await self.camera_repository.save(saved_camera)
                            logger.info(
                                f"Successfully registered camera {saved_camera.id} with Jetson backend "
                                f"and stored WebRTC config"
                            )
                        else:
                            logger.warning(
                                f"Camera {saved_camera.id} registered with Jetson backend but failed to get WebRTC config. "
                                f"Camera will work but WebRTC streaming may not be available."
                            )
                    else:
                        logger.warning(
                            f"Camera {saved_camera.id} saved locally but failed to register with Jetson backend. "
                            f"Camera will not be processed until registration succeeds."
                        )
                except httpx.TimeoutException as e:
                    logger.error(
                        f"Timeout while registering camera {saved_camera.id} with Jetson backend: {e}. "
                        f"Camera saved locally and will be synced when Jetson backend is available."
                    )
                    # Continue execution - camera is saved locally even if Jetson sync fails
                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"HTTP error registering camera {saved_camera.id} with Jetson backend: "
                        f"{e.response.status_code} - {e.response.text}. "
                        f"Camera saved locally."
                    )
                    # Continue execution - camera is saved locally even if Jetson sync fails
                except Exception as e:
                    logger.error(
                        f"Unexpected error registering camera {saved_camera.id} with Jetson backend: {e}",
                        exc_info=True
                    )
                    # Continue execution - camera is saved locally even if Jetson sync fails
            else:
                logger.info(
                    f"JetsonClient not available. Camera {saved_camera.id} saved locally only. "
                    f"Register JetsonClient to enable Jetson backend integration."
                )
        
        # Convert WebRTC config dict to DTO if available
        webrtc_config_dto = None
        if saved_camera.webrtc_config:
            try:
                webrtc_config_dto = WebRTCConfig(**saved_camera.webrtc_config)
            except Exception as e:
                logger.warning(f"Failed to parse WebRTC config for camera {saved_camera.id}: {e}")
        
        # Return DTO
        return CameraResponse(
            id=saved_camera.id or "",
            name=saved_camera.name,
            stream_url=saved_camera.stream_url,
            device_id=saved_camera.device_id,
            webrtc_config=webrtc_config_dto,
        )

