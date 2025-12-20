from typing import TYPE_CHECKING
from ...infrastructure.db.mongo_connection import (
    get_database,
    get_user_collection,
    get_camera_collection,
    get_agent_collection,
    get_device_collection,
)

if TYPE_CHECKING:
    from ..base_container import BaseContainer


class DatabaseProvider:
    """Centralized database connection provider - single source of truth for all DB connections"""
    
    @staticmethod
    def register(container: "BaseContainer") -> None:
        """
        Register all database collections in the container.
        This is the ONLY place where database connections are registered.
        Change database here, and all repositories automatically get the new connection.
        """
        # Get database instance
        database = get_database()
        
        # Register all collections as singletons
        # If you need to change database (MongoDB -> PostgreSQL), change only:
        # 1. mongo_connection.py -> postgres_connection.py
        # 2. This provider
        # All repositories will automatically use the new connection!
        
        container.register_singleton("database", database)
        container.register_singleton("user_collection", get_user_collection())
        container.register_singleton("camera_collection", get_camera_collection())
        container.register_singleton("agent_collection", get_agent_collection())
        container.register_singleton("device_collection", get_device_collection())
        
        # Future collections can be added here:
        # container.register_singleton("video_collection", get_video_collection())
        # container.register_singleton("detection_collection", get_detection_collection())

