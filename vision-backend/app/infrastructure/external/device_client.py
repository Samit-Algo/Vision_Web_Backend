# Standard library imports
import logging
from typing import Optional

# External package imports
import httpx

# Local application imports
from .base_jetson_client import BaseJetsonClient
from ...domain.constants import DeviceFields

logger = logging.getLogger(__name__)


class DeviceClient(BaseJetsonClient):
    """
    HTTP client for device-related communication with Jetson backend (vision_core).
    
    This client handles device registration with the Jetson device backend.
    """
    
    async def check_connection(self) -> bool:
        """
        Check if Jetson backend is reachable.
        
        Returns:
            True if connection successful, False otherwise
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
            except Exception as e:
                logger.debug(f"Connection check failed: {e}")
                return False
    
    async def register_device(
        self,
        device_id: str,
        web_backend_url: str,
        user_id: str,
        name: Optional[str] = None
    ) -> bool:
        """
        Register device with Jetson backend.
        
        This method sends device configuration to the Jetson backend so it can
        store the web backend URL and establish bidirectional communication.
        
        Args:
            device_id: Unique device ID
            web_backend_url: Web backend URL for this device
            user_id: User ID who owns this device
            name: Optional device name
            
        Returns:
            True if registration successful, False otherwise
            
        Note:
            This method logs errors but doesn't raise exceptions to prevent
            device creation from failing if Jetson backend is temporarily unavailable.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                payload = {
                    "device_id": device_id,  # Maps to DeviceFields.ID, but Jetson API expects "device_id"
                    "web_backend_url": web_backend_url,  # Not a Device field, Jetson API parameter
                    "user_id": user_id,  # Maps to DeviceFields.OWNER_USER_ID, but Jetson API expects "user_id"
                    DeviceFields.NAME: name  # Use constant for name field
                }
                
                logger.info(
                    f"Registering device {device_id} with Jetson backend at {self.base_url}"
                )
                
                response = await client.post(
                    f"{self.base_url}/api/v1/devices/create",
                    json=payload
                )
                response.raise_for_status()
                
                logger.info(f"Successfully registered device {device_id} with Jetson backend")
                return True
                
            except httpx.TimeoutException:
                logger.error(
                    f"Timeout while registering device {device_id} with Jetson backend"
                )
                return False
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error registering device {device_id} with Jetson backend: "
                    f"{e.response.status_code} - {e.response.text}"
                )
                return False
            except Exception as e:
                logger.error(
                    f"Unexpected error registering device {device_id} with Jetson backend: {e}",
                    exc_info=True
                )
                return False

