# Standard library imports
import secrets
import logging
from typing import Optional

# Local application imports
from ....domain.repositories.camera_repository import CameraRepository
from ....domain.repositories.user_repository import UserRepository
from ....domain.models.camera import Camera
from ...dto.camera_dto import CameraCreateRequest, CameraResponse

logger = logging.getLogger(__name__)


class CreateCameraUseCase:
    """Use case for creating a new camera"""

    def __init__(
        self,
        camera_repository: CameraRepository,
        user_repository: UserRepository,
    ) -> None:
        self.camera_repository = camera_repository
        self.user_repository = user_repository

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
        """
        # Generate camera ID
        camera_id = self._generate_camera_id()

        # Create domain camera entity
        new_camera = Camera(
            id=camera_id,
            owner_user_id=owner_user_id,
            name=request.name,
            stream_url=request.stream_url,
        )

        # Save camera
        saved_camera = await self.camera_repository.save(new_camera)

        # Return DTO
        return CameraResponse(
            id=saved_camera.id or "",
            name=saved_camera.name,
            stream_url=saved_camera.stream_url,
            webrtc_config=None,
        )
