from abc import ABC, abstractmethod
from typing import Optional
from ..models.device import Device


class DeviceRepository(ABC):
    """Repository interface - defines contract for device data access"""
    
    @abstractmethod
    async def find_by_id(self, device_id: str) -> Optional[Device]:
        """Find device by ID"""
        pass
    
    @abstractmethod
    async def find_by_owner(self, owner_user_id: str) -> list[Device]:
        """Find all devices owned by a user"""
        pass
    
    @abstractmethod
    async def save(self, device: Device) -> Device:
        """Save device (create or update)"""
        pass

