# Local application imports
from ....domain.repositories.camera_repository import CameraRepository
from ...dto.camera_dto import CameraResponse, WebRTCConfig


class GetCameraUseCase:
    """Use case for getting a camera by ID"""
    
    def __init__(self, camera_repository: CameraRepository) -> None:
        self.camera_repository = camera_repository
    
    async def execute(self, camera_id: str, owner_user_id: str) -> CameraResponse:
        """
        Get a camera by ID
        
        Args:
            camera_id: ID of the camera
            owner_user_id: ID of the user (for authorization check)
            
        Returns:
            CameraResponse with camera information
            
        Raises:
            ValueError: If camera not found or doesn't belong to user
        """
        camera = await self.camera_repository.find_by_id(camera_id)
        
        if camera is None:
            raise ValueError("Camera not found")
        
        if camera.owner_user_id != owner_user_id:
            raise ValueError("Camera not found")
        
        # Convert WebRTC config dict to DTO if available
        webrtc_config_dto = None
        if camera.webrtc_config:
            try:
                webrtc_config_dto = WebRTCConfig(**camera.webrtc_config)
            except Exception:
                # If parsing fails, just skip WebRTC config
                pass
        
        return CameraResponse(
            id=camera.id or "",
            name=camera.name,
            stream_url=camera.stream_url,
            device_id=camera.device_id,
            webrtc_config=webrtc_config_dto,
        )

