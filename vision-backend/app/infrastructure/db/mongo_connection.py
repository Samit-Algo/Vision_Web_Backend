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

    Uses explicit timeouts to fail fast instead of hanging 30+ seconds on unreachable MongoDB.
    On Windows, prefer 127.0.0.1 over localhost to avoid IPv6 resolution delays.

    Returns:
        MongoDB database instance
    """
    global _mongo_client, _mongo_database

    if _mongo_database is not None:
        return _mongo_database

    settings = get_settings()
    _mongo_client = AsyncIOMotorClient(
        settings.mongo_uri,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
    )
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


def get_event_collection() -> AsyncIOMotorCollection:
    """
    Get events collection from MongoDB

    Returns:
        MongoDB collection for events
    """
    return get_database()["events"]


def get_person_gallery_collection() -> AsyncIOMotorCollection:
    """
    Get person_gallery collection from MongoDB (reference photos + face embeddings).

    Returns:
        MongoDB collection for person_gallery
    """
    return get_database()["person_gallery"]