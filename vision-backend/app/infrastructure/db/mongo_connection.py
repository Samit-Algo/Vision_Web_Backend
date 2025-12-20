# Standard library imports
from typing import Optional

# External package imports
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

# Local application imports
from ...core.config import get_settings


# Global MongoDB connection instances (singleton pattern)
_mongo_client: Optional[AsyncIOMotorClient] = None
_mongo_database: Optional[AsyncIOMotorDatabase] = None


def get_database() -> AsyncIOMotorDatabase:
    """
    Get MongoDB database instance (singleton pattern)
    
    Returns:
        MongoDB database instance
    """
    global _mongo_client, _mongo_database
    
    if _mongo_database is not None:
        return _mongo_database
    
    settings = get_settings()
    _mongo_client = AsyncIOMotorClient(settings.mongo_uri)
    _mongo_database = _mongo_client[settings.mongo_database_name]
    return _mongo_database


def get_user_collection() -> AsyncIOMotorCollection:
    """
    Get users collection from MongoDB
    
    Returns:
        MongoDB collection for users
    """
    return get_database()["users"]


def get_camera_collection() -> AsyncIOMotorCollection:
    """
    Get cameras collection from MongoDB
    
    Returns:
        MongoDB collection for cameras
    """
    return get_database()["cameras"]


def get_agent_collection() -> AsyncIOMotorCollection:
    """
    Get agents collection from MongoDB
    
    Returns:
        MongoDB collection for agents
    """
    return get_database()["agents"]


def get_device_collection() -> AsyncIOMotorCollection:
    """
    Get devices collection from MongoDB
    
    Returns:
        MongoDB collection for devices
    """
    return get_database()["devices"]
