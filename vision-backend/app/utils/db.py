"""
Database utilities
------------------

Role:
- Provide a single entry-point to get MongoDB collections using PyMongo (sync).
- Used by runner and worker processes which run in separate threads/processes.
- Uses vision-backend's config structure and database name.

Note: This uses synchronous PyMongo (not Motor) because runner/worker run in
separate processes and use blocking operations. Vision-backend's main app uses
async Motor, but this is for the processing pipeline.
"""
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from ..core.config import get_settings

# Singleton client to avoid creating new connections on every call (was causing 1-2 min delays)
_mongo_client: Optional[MongoClient] = None
_mongo_db: Optional[Database] = None


def _get_client() -> MongoClient:
    """Get or create singleton MongoClient with connection timeouts (fail fast)."""
    global _mongo_client, _mongo_db
    if _mongo_client is not None:
        return _mongo_client

    settings = get_settings()
    mongo_uri = settings.mongo_uri
    if not mongo_uri:
        raise RuntimeError("❌ MONGO_URI not set. Please configure it in your .env file.")

    # Use explicit timeouts to fail fast instead of hanging 30+ seconds on unreachable MongoDB
    # On Windows, prefer 127.0.0.1 over localhost to avoid IPv6 resolution delays
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
    )
    try:
        client.admin.command("ping")
    except Exception as e:
        client.close()
        raise RuntimeError(f"❌ Failed to connect to MongoDB: {e}") from e
    _mongo_client = client
    return _mongo_client


def get_collection(collection_name: str = "agents") -> Collection:
    """
    Get MongoDB collection (singleton client, sync PyMongo).

    Uses vision-backend's configuration (MONGO_URI, MONGO_DB_NAME).

    Args:
        collection_name: Name of the collection (default: 'agents')

    Returns:
        MongoDB Collection object (synchronous PyMongo)
    """
    global _mongo_db
    client = _get_client()
    if _mongo_db is None:
        _mongo_db = client[get_settings().mongo_database_name]
    return _mongo_db[collection_name]
