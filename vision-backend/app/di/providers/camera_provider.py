from typing import TYPE_CHECKING
from ...domain.repositories.camera_repository import CameraRepository
from ...domain.repositories.user_repository import UserRepository
from ...domain.repositories.agent_repository import AgentRepository
from ...application.use_cases.camera.create_camera import CreateCameraUseCase
from ...application.use_cases.camera.list_cameras import ListCamerasUseCase
from ...application.use_cases.camera.get_camera import GetCameraUseCase
from ...application.use_cases.agent.list_agents_by_camera import ListAgentsByCameraUseCase

if TYPE_CHECKING:
    from ..base_container import BaseContainer


class CameraProvider:
    """Camera use case provider - registers all camera-related use cases"""
    
    @staticmethod
    def register(container: "BaseContainer") -> None:
        """
        Register all camera use cases.
        Use cases are created on-demand via factories.
        """
        # Register CreateCameraUseCase
        container.register_factory(
            CreateCameraUseCase,
            lambda: CreateCameraUseCase(
                camera_repository=container.get(CameraRepository),
                user_repository=container.get(UserRepository),
            )
        )
        
        # Register ListCamerasUseCase
        container.register_factory(
            ListCamerasUseCase,
            lambda: ListCamerasUseCase(
                camera_repository=container.get(CameraRepository)
            )
        )
        
        # Register GetCameraUseCase
        container.register_factory(
            GetCameraUseCase,
            lambda: GetCameraUseCase(
                camera_repository=container.get(CameraRepository)
            )
        )
        
        # Register ListAgentsByCameraUseCase
        container.register_factory(
            ListAgentsByCameraUseCase,
            lambda: ListAgentsByCameraUseCase(
                agent_repository=container.get(AgentRepository)
            )
        )

