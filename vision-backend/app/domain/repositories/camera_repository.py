from abc import ABC, abstractmethod
from typing import Optional, List
from ..models.camera import Camera


class CameraRepository(ABC):
    """Repository interface - defines contract for camera data access"""
    
    @abstractmethod
    async def find_by_id(self, camera_id: str) -> Optional[Camera]:
        """Find camera by ID"""
        pass
    
    @abstractmethod
    async def find_by_owner(self, owner_user_id: str) -> List[Camera]:
        """Find all cameras owned by a user"""
        pass
    
    @abstractmethod
    async def find_by_device(self, device_id: str) -> List[Camera]:
        """Find all cameras for a device"""
        pass
    
    @abstractmethod
    async def save(self, camera: Camera) -> Camera:
        """Save camera (create or update)"""
        pass

