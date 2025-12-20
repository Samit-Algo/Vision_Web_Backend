# Standard library imports
from typing import Optional, List, Dict, Any

# External package imports
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from bson.errors import InvalidId

# Local application imports
from ...domain.repositories.device_repository import DeviceRepository
from ...domain.models.device import Device
from .mongo_connection import get_device_collection


class MongoDeviceRepository(DeviceRepository):
    """MongoDB implementation of DeviceRepository"""
    
    def __init__(self, device_collection: Optional[AsyncIOMotorCollection] = None) -> None:
        self.device_collection = device_collection if device_collection is not None else get_device_collection()
    
    async def find_by_id(self, device_id: str) -> Optional[Device]:
        """Find device by ID"""
        if not device_id:
            return None
        
        try:
            try:
                object_id = ObjectId(device_id)
                document = await self.device_collection.find_one({"_id": object_id})
            except (InvalidId, ValueError, TypeError):
                document = await self.device_collection.find_one({"id": device_id})
            
            if document is None:
                return None
            
            return self._document_to_device(document)
        except Exception as e:
            raise RuntimeError(f"Error finding device by ID: {str(e)}")
    
    async def find_by_owner(self, owner_user_id: str) -> List[Device]:
        """Find all devices owned by a user"""
        if not owner_user_id:
            return []
        
        try:
            cursor = self.device_collection.find({"owner_user_id": owner_user_id})
            devices = []
            async for document in cursor:
                devices.append(self._document_to_device(document))
            return devices
        except Exception as e:
            raise RuntimeError(f"Error listing devices for owner: {str(e)}")
    
    async def save(self, device: Device) -> Device:
        """Save device (create new or update existing)"""
        if not device:
            raise ValueError("Device cannot be None")
        
        try:
            device_dict = self._device_to_dict(device)
            
            if device.id:
                try:
                    object_id = ObjectId(device.id)
                    update_result = await self.device_collection.update_one(
                        {"_id": object_id},
                        {"$set": {k: v for k, v in device_dict.items() if k != "_id"}}
                    )
                    if update_result.matched_count > 0:
                        updated_document = await self.device_collection.find_one({"_id": object_id})
                        if updated_document:
                            return self._document_to_device(updated_document)
                except (InvalidId, ValueError, TypeError):
                    update_result = await self.device_collection.update_one(
                        {"id": device.id},
                        {"$set": {k: v for k, v in device_dict.items() if k not in ["_id", "id"]}}
                    )
                    if update_result.matched_count > 0:
                        updated_document = await self.device_collection.find_one({"id": device.id})
                        if updated_document:
                            return self._document_to_device(updated_document)
                
                if "_id" in device_dict:
                    del device_dict["_id"]
            
            result = await self.device_collection.insert_one(device_dict)
            new_document = await self.device_collection.find_one({"_id": result.inserted_id})
            if new_document is None:
                raise RuntimeError("Device was created but could not be retrieved")
            
            return self._document_to_device(new_document)
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error saving device: {str(e)}")
    
    def _document_to_device(self, document: Dict[str, Any]) -> Device:
        """Convert MongoDB document to Device domain model"""
        if not document:
            raise ValueError("Invalid document: document is None or empty")
        
        device_id = None
        if "id" in document:
            device_id = document["id"]
        elif "_id" in document:
            device_id = str(document["_id"])
        
        return Device(
            id=device_id,
            owner_user_id=document.get("owner_user_id", ""),
            name=document.get("name", ""),
            jetson_backend_url=document.get("jetson_backend_url", ""),
        )
    
    def _device_to_dict(self, device: Device) -> Dict[str, Any]:
        """Convert Device domain model to MongoDB document"""
        if not device:
            raise ValueError("Device cannot be None")
        
        device_dict: Dict[str, Any] = {
            "owner_user_id": device.owner_user_id,
            "name": device.name,
            "jetson_backend_url": device.jetson_backend_url,
        }
        
        if device.id:
            try:
                device_dict["_id"] = ObjectId(device.id)
            except (InvalidId, ValueError, TypeError):
                device_dict["id"] = device.id
        
        return device_dict

