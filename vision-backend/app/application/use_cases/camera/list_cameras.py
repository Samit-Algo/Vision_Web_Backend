# Standard library imports
from typing import List

# Local application imports
from ....domain.repositories.camera_repository import CameraRepository
from ...dto.camera_dto import CameraResponse, WebRTCConfig


class ListCamerasUseCase:
    """Use case for listing cameras for a user"""
    
    def __init__(self, camera_repository: CameraRepository) -> None:
        self.camera_repository = camera_repository
    
    async def execute(self, owner_user_id: str) -> List[CameraResponse]:
        """
        List all cameras owned by a user
        
        Args:
            owner_user_id: ID of the user
            
        Returns:
            List of CameraResponse objects
        """
        cameras = await self.camera_repository.find_by_owner(owner_user_id)
        
        result = []
        for camera in cameras:
            # Convert WebRTC config dict to DTO if available
            webrtc_config_dto = None
            if camera.webrtc_config:
                try:
                    webrtc_config_dto = WebRTCConfig(**camera.webrtc_config)
                except Exception:
                    # If parsing fails, just skip WebRTC config
                    pass
            
            result.append(
                CameraResponse(
                    id=camera.id or "",
                    name=camera.name,
                    stream_url=camera.stream_url,
                    device_id=camera.device_id,
                    stream_config=camera.stream_config,
                    webrtc_config=webrtc_config_dto,
                )
            )
        
        return result

