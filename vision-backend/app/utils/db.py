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
from pymongo import MongoClient
from pymongo.collection import Collection
from app.core.config import get_settings


def get_collection(collection_name: str = "agents") -> Collection:
    """
    Connect to MongoDB and return a collection (synchronous PyMongo).
    
    Uses vision-backend's configuration (MONGO_URI, MONGO_DB_NAME).
    
    Args:
        collection_name: Name of the collection (default: 'agents' to match vision-backend convention)
    
    Returns:
        MongoDB Collection object (synchronous PyMongo)
    """
    settings = get_settings()
    mongo_uri = settings.mongo_uri
    db_name = settings.mongo_database_name
    
    if not mongo_uri:
        raise RuntimeError("❌ MONGO_URI not set. Please configure it in your .env file.")
    
    client = MongoClient(mongo_uri)
    collection = client[db_name][collection_name]
    
    # Verify connection
    try:
        client.admin.command('ping')
    except Exception as e:
        raise RuntimeError(f"❌ Failed to connect to MongoDB: {e}")
    
    return collection
