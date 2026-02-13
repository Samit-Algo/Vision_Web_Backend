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
    async def search_by_name(self, query: str, owner_user_id: str, limit: int = 5) -> List[Camera]:
        """
        Search cameras by name using regex (case-insensitive partial matching).
        
        Args:
            query: Search query (will be used for regex matching)
            owner_user_id: The owner user ID to filter by
            limit: Maximum number of results to return (default: 5)
            
        Returns:
            List of Camera domain models matching the query
        """
        pass
    
    @abstractmethod
    async def save(self, camera: Camera) -> Camera:
        """Save camera (create or update)"""
        pass

