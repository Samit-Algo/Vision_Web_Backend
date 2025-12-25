# Standard library imports
import logging
from typing import Optional, Dict, Any

# External package imports
import httpx

# Local application imports
from .base_jetson_client import BaseJetsonClient
from ...domain.constants import CameraFields

logger = logging.getLogger(__name__)


class CameraClient(BaseJetsonClient):
    """
    HTTP client for camera-related communication with Jetson backend (vision_core).
    
    This client handles camera registration and WebRTC configuration retrieval
    for cameras.
    """
    
    async def register_camera(
        self,
        camera_id: str,
        owner_user_id: str,
        name: str,
        stream_url: str,
        device_id: Optional[str] = None
    ) -> bool:
        """
        Register camera with Jetson backend.
        
        This method sends camera configuration to the Jetson device so it can
        start processing the RTSP stream and make it available via WebRTC.
        
        Uses same field names as web backend (no mapping/conversion).
        
        Args:
            camera_id: Unique camera ID (e.g., "CAM-43C1E6AFB726")
            owner_user_id: User ID who owns this camera
            name: Camera name
            stream_url: Stream URL
            device_id: Optional device ID
            
        Returns:
            True if registration successful, False otherwise
            
        Note:
            This method logs errors but doesn't raise exceptions to prevent
            camera creation from failing if Jetson backend is temporarily unavailable.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Use same field names as web backend (no mapping)
                payload = {
                    CameraFields.ID: camera_id,
                    CameraFields.OWNER_USER_ID: owner_user_id,
                    CameraFields.NAME: name,
                    CameraFields.STREAM_URL: stream_url,
                    CameraFields.DEVICE_ID: device_id
                }
                
                logger.info(
                    f"Registering camera {camera_id} with Jetson backend at {self.base_url}"
                )
                
                response = await client.post(
                    f"{self.base_url}/api/v1/cameras/create",
                    json=payload
                )
                response.raise_for_status()
                
                logger.info(f"Successfully registered camera {camera_id} with Jetson backend")
                return True
                
            except httpx.TimeoutException:
                logger.error(
                    f"Timeout while registering camera {camera_id} with Jetson backend"
                )
                return False
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error registering camera {camera_id} with Jetson backend: "
                    f"{e.response.status_code} - {e.response.text}"
                )
                return False
            except Exception as e:
                logger.error(
                    f"Unexpected error registering camera {camera_id} with Jetson backend: {e}",
                    exc_info=True
                )
                return False
    
    async def get_webrtc_config(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get WebRTC configuration from Jetson backend.
        
        This retrieves the signaling server URL and ICE servers needed for
        the frontend to establish WebRTC connections.
        
        Args:
            user_id: User ID to get configuration for
            
        Returns:
            Dictionary with 'signaling_url' and 'ice_servers' keys, or None if error
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                logger.info(f"Fetching WebRTC config for user {user_id} from Jetson backend")
                
                response = await client.get(
                    f"{self.base_url}/api/v1/cameras/stream-config",
                    params={"user_id": user_id}
                )
                response.raise_for_status()
                
                config = response.json()
                logger.info(f"Successfully retrieved WebRTC config for user {user_id}")
                return config
                
            except httpx.TimeoutException:
                logger.error(
                    f"Timeout while fetching WebRTC config for user {user_id} from Jetson backend"
                )
                return None
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error fetching WebRTC config for user {user_id}: "
                    f"{e.response.status_code} - {e.response.text}"
                )
                return None
            except Exception as e:
                logger.error(
                    f"Unexpected error fetching WebRTC config for user {user_id}: {e}",
                    exc_info=True
                )
                return None
    
    async def get_webrtc_config_for_camera(
        self, 
        user_id: str, 
        camera_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get WebRTC configuration for a specific camera.
        
        This retrieves camera-specific WebRTC config from Jetson backend.
        The signaling URL includes the camera_id for per-camera streaming.
        
        Args:
            user_id: User ID who owns the camera
            camera_id: Camera ID to get configuration for
            
        Returns:
            Dictionary with 'signaling_url', 'viewer_id', and 'ice_servers' keys, or None if error
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                logger.info(
                    f"Fetching camera-specific WebRTC config for camera {camera_id} "
                    f"(user {user_id}) from Jetson backend"
                )
                
                response = await client.get(
                    f"{self.base_url}/api/v1/cameras/{camera_id}/stream-config",
                    params={"user_id": user_id}
                )
                response.raise_for_status()
                
                config = response.json()
                
                # Add viewer_id for compatibility (derived from signaling_url)
                signaling_url = config.get("signaling_url", "")
                viewer_id = None
                
                # Extract viewer_id from signaling_url
                # Jetson backend format (from get_camera_stream_config.py):
                # AWS: ws://aws-url:8000/ws/viewer:{user_id}:{camera_id} -> viewer_id: viewer:{user_id}:{camera_id}
                # Local: ws://localhost:8765/viewer:{user_id}/{camera_id} -> viewer_id: viewer:{user_id}/{camera_id}
                if signaling_url:
                    # Check for AWS format first (colon separator)
                    if f"viewer:{user_id}:{camera_id}" in signaling_url:
                        viewer_id = f"viewer:{user_id}:{camera_id}"
                    # Check for local format (slash separator)
                    elif f"viewer:{user_id}/{camera_id}" in signaling_url:
                        viewer_id = f"viewer:{user_id}/{camera_id}"
                    # Fallback: construct from known format (prefer AWS format)
                    else:
                        viewer_id = f"viewer:{user_id}:{camera_id}"
                        logger.warning(
                            f"Could not extract viewer_id from signaling_url format. "
                            f"Using constructed format: {viewer_id}"
                        )
                
                config["viewer_id"] = viewer_id
                
                logger.info(
                    f"Successfully retrieved camera-specific WebRTC config for camera {camera_id}"
                )
                return config
                
            except httpx.TimeoutException:
                logger.error(
                    f"Timeout while fetching camera-specific WebRTC config for camera {camera_id} "
                    f"from Jetson backend"
                )
                return None
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error fetching camera-specific WebRTC config for camera {camera_id}: "
                    f"{e.response.status_code} - {e.response.text}"
                )
                return None
            except Exception as e:
                logger.error(
                    f"Unexpected error fetching camera-specific WebRTC config for camera {camera_id}: {e}",
                    exc_info=True
                )
                return None

