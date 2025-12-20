from typing import TYPE_CHECKING
from ...domain.repositories.user_repository import UserRepository
from ...domain.repositories.camera_repository import CameraRepository
from ...domain.repositories.agent_repository import AgentRepository
from ...domain.repositories.device_repository import DeviceRepository
from ...infrastructure.db.mongo_user_repository import MongoUserRepository
from ...infrastructure.db.mongo_camera_repository import MongoCameraRepository
from ...infrastructure.db.mongo_agent_repository import MongoAgentRepository
from ...infrastructure.db.mongo_device_repository import MongoDeviceRepository

if TYPE_CHECKING:
    from ..base_container import BaseContainer


class RepositoryProvider:
    """Repository registration provider - wires domain interfaces to infrastructure implementations"""
    
    @staticmethod
    def register(container: "BaseContainer") -> None:
        """
        Register all repository implementations.
        Gets collections from database provider and creates repository instances.
        """
        # Get collections from database provider
        user_collection = container.get("user_collection")
        camera_collection = container.get("camera_collection")
        agent_collection = container.get("agent_collection")
        device_collection = container.get("device_collection")
        
        # Register repository implementations
        # Domain interfaces -> Infrastructure implementations
        container.register_singleton(
            UserRepository,
            MongoUserRepository(user_collection=user_collection)
        )
        
        container.register_singleton(
            CameraRepository,
            MongoCameraRepository(camera_collection=camera_collection)
        )
        
        container.register_singleton(
            AgentRepository,
            MongoAgentRepository(agent_collection=agent_collection)
        )
        
        container.register_singleton(
            DeviceRepository,
            MongoDeviceRepository(device_collection=device_collection)
        )

