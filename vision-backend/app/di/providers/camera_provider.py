from typing import TYPE_CHECKING
from ...domain.repositories.camera_repository import CameraRepository
from ...domain.repositories.user_repository import UserRepository
from ...domain.repositories.device_repository import DeviceRepository
from ...domain.repositories.agent_repository import AgentRepository
from ...application.use_cases.camera.create_camera import CreateCameraUseCase
from ...application.use_cases.camera.list_cameras import ListCamerasUseCase
from ...application.use_cases.camera.get_camera import GetCameraUseCase
from ...application.use_cases.agent.list_agents_by_camera import ListAgentsByCameraUseCase
from ...infrastructure.external.camera_client import CameraClient

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
        # Register CameraClient as singleton (shared across all use cases)
        # Check if already registered to avoid creating multiple instances
        try:
            container.get(CameraClient)
        except ValueError:
            # Not registered yet, register it now
            camera_client = CameraClient()
            container.register_singleton(CameraClient, camera_client)
        
        # Register CreateCameraUseCase
        container.register_factory(
            CreateCameraUseCase,
            lambda: CreateCameraUseCase(
                camera_repository=container.get(CameraRepository),
                user_repository=container.get(UserRepository),
                device_repository=container.get(DeviceRepository),
                jetson_client=container.get(CameraClient)
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

