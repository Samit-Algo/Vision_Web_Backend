# Standard library imports
from typing import Optional, List, Dict, Any

# External package imports
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from bson.errors import InvalidId

# Local application imports
from ...domain.repositories.camera_repository import CameraRepository
from ...domain.models.camera import Camera
from ...domain.constants import CameraFields
from .mongo_connection import get_camera_collection


class MongoCameraRepository(CameraRepository):
    """MongoDB implementation of CameraRepository"""
    
    def __init__(self, camera_collection: Optional[AsyncIOMotorCollection] = None) -> None:
        self.camera_collection = camera_collection if camera_collection is not None else get_camera_collection()
    
    async def find_by_id(self, camera_id: str) -> Optional[Camera]:
        """
        Find camera by ID
        
        Args:
            camera_id: The camera ID to find
            
        Returns:
            Camera domain model if found, None otherwise
        """
        if not camera_id:
            return None
        
        try:
            # Try to convert to ObjectId if it's a MongoDB ObjectId string
            try:
                object_id = ObjectId(camera_id)
                document = await self.camera_collection.find_one({CameraFields.MONGO_ID: object_id})
            except (InvalidId, ValueError, TypeError):
                # If not ObjectId, try searching by "id" field (for custom IDs)
                document = await self.camera_collection.find_one({CameraFields.ID: camera_id})
            
            if document is None:
                return None
            
            return self._document_to_camera(document)
        except Exception as e:
            raise RuntimeError(f"Error finding camera by ID: {str(e)}")
    
    async def find_by_owner(self, owner_user_id: str) -> List[Camera]:
        """
        Find all cameras owned by a user
        
        Args:
            owner_user_id: The owner user ID
            
        Returns:
            List of Camera domain models
        """
        if not owner_user_id:
            return []
        
        try:
            cursor = self.camera_collection.find({CameraFields.OWNER_USER_ID: owner_user_id})
            cameras = []
            async for document in cursor:
                cameras.append(self._document_to_camera(document))
            return cameras
        except Exception as e:
            raise RuntimeError(f"Error listing cameras for owner: {str(e)}")
    
    async def find_by_device(self, device_id: str) -> List[Camera]:
        """
        Find all cameras for a device
        
        Args:
            device_id: The device ID
            
        Returns:
            List of Camera domain models
        """
        if not device_id:
            return []
        
        try:
            cursor = self.camera_collection.find({CameraFields.DEVICE_ID: device_id})
            cameras = []
            async for document in cursor:
                cameras.append(self._document_to_camera(document))
            return cameras
        except Exception as e:
            raise RuntimeError(f"Error listing cameras for device: {str(e)}")
    
    async def save(self, camera: Camera) -> Camera:
        """
        Save camera (create new or update existing)
        
        Args:
            camera: Camera domain model to save
            
        Returns:
            Saved Camera domain model with ID set
        """
        if not camera:
            raise ValueError("Camera cannot be None")
        
        try:
            camera_dict = self._camera_to_dict(camera)
            
            if camera.id:
                # Try to update existing camera
                try:
                    # Try ObjectId first
                    object_id = ObjectId(camera.id)
                    update_result = await self.camera_collection.update_one(
                        {CameraFields.MONGO_ID: object_id},
                        {"$set": {k: v for k, v in camera_dict.items() if k != CameraFields.MONGO_ID}}
                    )
                    if update_result.matched_count > 0:
                        # Fetch and return updated document
                        updated_document = await self.camera_collection.find_one({CameraFields.MONGO_ID: object_id})
                        if updated_document:
                            return self._document_to_camera(updated_document)
                except (InvalidId, ValueError, TypeError):
                    # If not ObjectId, try updating by "id" field
                    update_result = await self.camera_collection.update_one(
                        {CameraFields.ID: camera.id},
                        {"$set": {k: v for k, v in camera_dict.items() if k not in [CameraFields.MONGO_ID, CameraFields.ID]}}
                    )
                    if update_result.matched_count > 0:
                        # Fetch and return updated document
                        updated_document = await self.camera_collection.find_one({CameraFields.ID: camera.id})
                        if updated_document:
                            return self._document_to_camera(updated_document)
                
                # If update didn't match, create new with provided ID
                if CameraFields.MONGO_ID in camera_dict:
                    del camera_dict[CameraFields.MONGO_ID]
            
            # Create new camera
            result = await self.camera_collection.insert_one(camera_dict)
            
            # Fetch and return the newly created document
            new_document = await self.camera_collection.find_one({CameraFields.MONGO_ID: result.inserted_id})
            if new_document is None:
                raise RuntimeError("Camera was created but could not be retrieved")
            
            return self._document_to_camera(new_document)
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error saving camera: {str(e)}")
    
    def _document_to_camera(self, document: Dict[str, Any]) -> Camera:
        """
        Convert MongoDB document to Camera domain model
        
        Args:
            document: MongoDB document dictionary
            
        Returns:
            Camera domain model
        """
        if not document:
            raise ValueError("Invalid document: document is None or empty")
        
        # Extract ID - prioritize custom "id" field over MongoDB "_id"
        # This ensures we use the generated unique ID (e.g., "CAM-43C1E6AFB726") 
        # instead of the MongoDB ObjectId
        camera_id = None
        if CameraFields.ID in document:
            camera_id = document[CameraFields.ID]
        elif CameraFields.MONGO_ID in document:
            # Fallback to _id only if custom "id" field doesn't exist
            camera_id = str(document[CameraFields.MONGO_ID])
        
        return Camera(
            id=camera_id,
            owner_user_id=document.get(CameraFields.OWNER_USER_ID, ""),
            name=document.get(CameraFields.NAME, ""),
            stream_url=document.get(CameraFields.STREAM_URL, ""),
            device_id=document.get(CameraFields.DEVICE_ID),
            stream_config=document.get(CameraFields.STREAM_CONFIG),  # Keep as dict, no conversion needed
            webrtc_config=document.get(CameraFields.WEBRTC_CONFIG),  # Keep as dict, no conversion needed
        )
    
    def _camera_to_dict(self, camera: Camera) -> Dict[str, Any]:
        """
        Convert Camera domain model to MongoDB document
        
        Args:
            camera: Camera domain model
            
        Returns:
            Dictionary ready for MongoDB storage
        """
        if not camera:
            raise ValueError("Camera cannot be None")
        
        camera_dict: Dict[str, Any] = {
            CameraFields.OWNER_USER_ID: camera.owner_user_id,
            CameraFields.NAME: camera.name,
            CameraFields.STREAM_URL: camera.stream_url,
            CameraFields.DEVICE_ID: camera.device_id,
            CameraFields.STREAM_CONFIG: camera.stream_config,
            CameraFields.WEBRTC_CONFIG: camera.webrtc_config,
        }
        
        # Handle ID - if it's a MongoDB ObjectId string, convert it
        if camera.id:
            try:
                camera_dict[CameraFields.MONGO_ID] = ObjectId(camera.id)
            except (InvalidId, ValueError, TypeError):
                # If not a valid ObjectId, store as custom "id" field
                camera_dict[CameraFields.ID] = camera.id
        
        return camera_dict

