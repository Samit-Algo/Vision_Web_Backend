# Standard library imports
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

# External package imports
import httpx

# Local application imports
from .base_jetson_client import BaseJetsonClient
from ...domain.constants import AgentFields

logger = logging.getLogger(__name__)


class AgentClient(BaseJetsonClient):
    """
    HTTP client for agent-related communication with Jetson backend (vision_core).
    
    This client handles agent registration and stream configuration retrieval
    for agents.
    """
    
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
                    "agent_id": agent_id,  # Maps to AgentFields.ID, but Jetson API expects "agent_id"
                    "task_name": task_name,  # Not an Agent field, Jetson API parameter
                    "task_type": task_type,  # Not an Agent field, Jetson API parameter
                    AgentFields.CAMERA_ID: camera_id,
                    "source_uri": source_uri,  # Not an Agent field, Jetson API parameter
                    "model_ids": model_ids,  # Not an Agent field, Jetson API parameter (plural)
                    AgentFields.FPS: fps,
                    AgentFields.RUN_MODE: run_mode,
                    AgentFields.RULES: [rule if isinstance(rule, dict) else rule.dict() if hasattr(rule, 'dict') else rule for rule in rules],
                    AgentFields.STATUS: status,
                    "start_at": start_at_str,  # Maps to AgentFields.START_TIME, but Jetson API expects "start_at"
                    "end_at": end_at_str,  # Maps to AgentFields.END_TIME, but Jetson API expects "end_at"
                }
                
                logger.info(
                    f"Registering agent {agent_id} for camera {camera_id} with Jetson backend at {self.base_url}"
                )
                print(payload)
                
                response = await client.post(
                    f"{self.base_url}/api/v1/agents/create",
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
                agent_id = agent_config.get(AgentFields.ID, "")
                
                logger.info(
                    f"Registering agent {agent_id} with Jetson backend at {self.base_url} "
                    f"(using raw agent config from database)"
                )
                
                response = await client.post(
                    f"{self.base_url}/api/v1/agents/create",
                    json=agent_config
                )
                response.raise_for_status()
                
                logger.info(f"Successfully registered agent {agent_id} with Jetson backend")
                return True
                
            except httpx.TimeoutException:
                logger.error(
                    f"Timeout while registering agent {agent_config.get(AgentFields.ID, 'unknown')} with Jetson backend"
                )
                return False
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error registering agent {agent_config.get(AgentFields.ID, 'unknown')} with Jetson backend: "
                    f"{e.response.status_code} - {e.response.text}"
                )
                return False
            except Exception as e:
                logger.error(
                    f"Unexpected error registering agent {agent_config.get(AgentFields.ID, 'unknown')} with Jetson backend: {e}",
                    exc_info=True
                )
                return False
    
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
                    f"{self.base_url}/api/v1/agents/{agent_id}/stream-config"
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

