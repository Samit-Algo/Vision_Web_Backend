# Standard library imports
import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

# External package imports
import httpx

# Local application imports
from ...core.config import get_settings

logger = logging.getLogger(__name__)


class JetsonClient:
    """
    HTTP client for communicating with Jetson backend (vision_core).
    
    This client handles all communication with the Jetson device backend,
    including camera registration and WebRTC configuration retrieval.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        """
        Initialize Jetson client.
        
        Args:
            base_url: Base URL for Jetson backend. If None, reads from env.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.jetson_backend_url
        self.timeout = timeout
    
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
                    "device_id": device_id,
                    "web_backend_url": web_backend_url,
                    "user_id": user_id,
                    "name": name
                }
                
                logger.info(
                    f"Registering device {device_id} with Jetson backend at {self.base_url}"
                )
                
                response = await client.post(
                    f"{self.base_url}/api/devices",
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
                    "id": camera_id,
                    "owner_user_id": owner_user_id,
                    "name": name,
                    "stream_url": stream_url,
                    "device_id": device_id
                }
                
                logger.info(
                    f"Registering camera {camera_id} with Jetson backend at {self.base_url}"
                )
                
                response = await client.post(
                    f"{self.base_url}/api/cameras",
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
                    f"{self.base_url}/api/stream-config",
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
                    f"{self.base_url}/api/cameras/{camera_id}/stream-config",
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
    
    async def get_stream_config_for_agent(
        self, 
        agent_id: str,
        camera_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get stream configuration for a specific agent.
        
        This retrieves agent-specific stream config from Jetson backend.
        The stream includes bounding boxes and agent-specific annotations.
        
        Args:
            agent_id: Agent ID to get stream configuration for
            camera_id: Camera ID which the agent is monitoring
            user_id: User ID who owns the agent
            
        Returns:
            Dictionary with stream configuration (signaling_url, viewer_id, ice_servers, etc.), 
            or None if error
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                logger.info(
                    f"Fetching stream config for agent {agent_id} "
                    f"(user {user_id}) for camera {camera_id} from Jetson backend"
                )
                
                response = await client.get(
                    f"{self.base_url}/api/agents/{agent_id}/stream-config"
                )
                response.raise_for_status()
                
                config = response.json()
                
                # Add viewer_id for compatibility (derived from signaling_url)
                signaling_url = config.get("signaling_url", "")
                viewer_id = None
                
                # Extract viewer_id from signaling_url
                # Jetson backend format (similar to camera):
                # AWS: ws://aws-url:8000/ws/viewer:{user_id}:{camera_id}:{agent_id} -> viewer_id: viewer:{user_id}:{camera_id}:{agent_id}
                # Local: ws://localhost:8765/viewer:{user_id}/{camera_id}/{agent_id} -> viewer_id: viewer:{user_id}/{camera_id}/{agent_id}
                if signaling_url:
                    # Check for AWS format first (colon separator)
                    if f"viewer:{user_id}:{camera_id}:{agent_id}" in signaling_url:
                        viewer_id = f"viewer:{user_id}:{camera_id}:{agent_id}"
                    # Check for local format (slash separator)
                    elif f"viewer:{user_id}/{camera_id}/{agent_id}" in signaling_url:
                        viewer_id = f"viewer:{user_id}/{camera_id}/{agent_id}"
                    # Fallback: construct from known format (prefer AWS format)
                    else:
                        viewer_id = f"viewer:{user_id}:{camera_id}:{agent_id}"
                        logger.warning(
                            f"Could not extract viewer_id from signaling_url format for agent {agent_id} for camera {camera_id}. "
                            f"Using constructed format: {viewer_id}"
                        )
                
                config["viewer_id"] = viewer_id
                
                logger.info(
                    f"Successfully retrieved stream config for agent {agent_id} for camera {camera_id}"
                )
                return config
                
            except httpx.TimeoutException:
                logger.error(
                    f"Timeout while fetching stream config for agent {agent_id} for camera {camera_id} from Jetson backend"
                )
                return None
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error fetching stream config for agent {agent_id} for camera {camera_id}: "
                    f"{e.response.status_code} - {e.response.text}"
                )
                return None
            except Exception as e:
                logger.error(
                    f"Unexpected error fetching stream config for agent {agent_id} for camera {camera_id}: {e}",
                    exc_info=True
                )
                return None
    
    async def register_agent(
        self,
        agent_id: str,
        task_name: str,
        task_type: str,
        camera_id: str,
        source_uri: str,
        model_ids: List[str],
        fps: int,
        run_mode: str,
        rules: List[Dict[str, Any]],
        status: str,
        start_at: datetime,
        end_at: datetime,
    ) -> bool:
        """
        Register agent with Jetson backend.
        
        This method sends agent configuration to the Jetson device so it can
        start processing the camera stream according to the agent's rules.
        
        Args:
            agent_id: Unique agent ID
            task_name: Name of the task (e.g., "Person Detection")
            task_type: Type of task (e.g., "object_detection")
            camera_id: Camera ID this agent monitors
            source_uri: RTSP stream URL
            model_ids: List of AI model IDs to use
            fps: Frames per second to process
            run_mode: Agent run mode
            rules: List of detection rules
            status: Agent status (pending, running, etc.)
            start_at: When to start the agent
            end_at: When to stop the agent
            
        Returns:
            True if registration successful, False otherwise
            
        Note:
            This method logs errors but doesn't raise exceptions to prevent
            agent creation from failing if Jetson backend is temporarily unavailable.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Convert datetime objects to ISO format strings
                start_at_str = start_at.isoformat() if isinstance(start_at, datetime) else start_at
                end_at_str = end_at.isoformat() if isinstance(end_at, datetime) else end_at
                
                payload = {
                    "agent_id": agent_id,
                    "task_name": task_name,
                    "task_type": task_type,
                    "camera_id": camera_id,
                    "source_uri": source_uri,
                    "model_ids": model_ids,
                    "fps": fps,
                    "run_mode": run_mode,
                    "rules": [rule if isinstance(rule, dict) else rule.dict() if hasattr(rule, 'dict') else rule for rule in rules],
                    "status": status,
                    "start_at": start_at_str,
                    "end_at": end_at_str,
                }
                
                logger.info(
                    f"Registering agent {agent_id} for camera {camera_id} with Jetson backend at {self.base_url}"
                )
                print(payload)
                
                response = await client.post(
                    f"{self.base_url}/api/agents",
                    json=payload
                )
                response.raise_for_status()
                
                logger.info(f"Successfully registered agent {agent_id} with Jetson backend")
                return True
                
            except httpx.TimeoutException:
                logger.error(
                    f"Timeout while registering agent {agent_id} with Jetson backend"
                )
                return False
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error registering agent {agent_id} with Jetson backend: "
                    f"{e.response.status_code} - {e.response.text}"
                )
                return False
            except Exception as e:
                logger.error(
                    f"Unexpected error registering agent {agent_id} with Jetson backend: {e}",
                    exc_info=True
                )
                return False
    
    async def register_agent_raw(self, agent_config: Dict[str, Any]) -> bool:
        """
        Register agent with Jetson backend using raw agent config from database.
        
        This method sends the agent configuration exactly as stored in the database
        to the Jetson backend without any format conversion.
        
        Args:
            agent_config: Dictionary containing agent configuration (same format as database)
            
        Returns:
            True if registration successful, False otherwise
            
        Note:
            This method logs errors but doesn't raise exceptions to prevent
            agent creation from failing if Jetson backend is temporarily unavailable.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                agent_id = agent_config.get("id", "")
                
                logger.info(
                    f"Registering agent {agent_id} with Jetson backend at {self.base_url} "
                    f"(using raw agent config from database)"
                )
                
                response = await client.post(
                    f"{self.base_url}/api/agents",
                    json=agent_config
                )
                response.raise_for_status()
                
                logger.info(f"Successfully registered agent {agent_id} with Jetson backend")
                return True
                
            except httpx.TimeoutException:
                logger.error(
                    f"Timeout while registering agent {agent_config.get('id', 'unknown')} with Jetson backend"
                )
                return False
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error registering agent {agent_config.get('id', 'unknown')} with Jetson backend: "
                    f"{e.response.status_code} - {e.response.text}"
                )
                return False
            except Exception as e:
                logger.error(
                    f"Unexpected error registering agent {agent_config.get('id', 'unknown')} with Jetson backend: {e}",
                    exc_info=True
                )
                return False

