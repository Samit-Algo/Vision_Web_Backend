from __future__ import annotations

from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from ..session_state.agent_state import get_agent_state, reset_agent_state
from ...domain.models.agent import Agent

# Enable nested event loops to allow asyncio.run() from within async contexts
import nest_asyncio
nest_asyncio.apply()


# Global repository instances - will be set via dependency injection
_agent_repository: Optional[Any] = None
_camera_repository: Optional[Any] = None
_device_repository: Optional[Any] = None
_jetson_client: Optional[Any] = None


def set_agent_repository(repository):
    """Set the agent repository for saving agents"""
    global _agent_repository
    _agent_repository = repository


def set_camera_repository(repository):
    """Set the camera repository for looking up cameras"""
    global _camera_repository
    _camera_repository = repository


def set_device_repository(repository):
    """Set the device repository for looking up devices"""
    global _device_repository
    _device_repository = repository


def set_jetson_client(client):
    """Set the Jetson client for registering agents"""
    global _jetson_client
    _jetson_client = client


def save_to_db(session_id: str = "default", user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Finalize the agent configuration and save to database.

    Args:
        session_id: Session identifier for state management
        user_id: User ID who owns this agent

    Behavior:
    - Validate that the agent is in CONFIRMATION state with no missing fields.
    - Create Agent domain model.
    - Persist via agent repository.
    - Reset the agent state after saving.

    Returns a small summary for the calling agent.
    """
    if _agent_repository is None:
        raise ValueError("Agent repository not initialized. Call set_agent_repository() first.")

    agent_state = get_agent_state(session_id)

    if agent_state.status != "CONFIRMATION":
        raise ValueError("Cannot save: agent is not in CONFIRMATION state.")

    if agent_state.missing_fields:
        raise ValueError(f"Cannot save: missing fields {agent_state.missing_fields}")

    # Validate user_id is provided
    if not user_id:
        raise ValueError("Cannot save: user_id is required. Agent must be associated with an authenticated user.")

    # Validate run_mode
    run_mode = agent_state.fields.get("run_mode")
    if run_mode and run_mode not in ["continuous", "patrol"]:
        raise ValueError(f"Invalid run_mode: {run_mode}. Only 'continuous' or 'patrol' are allowed.")
    
    # Load rule to check requires_zone
    from .initialize_state_tool import _get_rule
    rule = _get_rule(agent_state.rule_id) if agent_state.rule_id else None
    requires_zone = bool(rule.get("requires_zone", False)) if rule else agent_state.fields.get("requires_zone", False)

    # Convert string times to datetime objects
    start_time_str = agent_state.fields.get("start_time")
    end_time_str = agent_state.fields.get("end_time")
    start_time = None
    end_time = None
    
    if start_time_str:
        if isinstance(start_time_str, str):
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        elif isinstance(start_time_str, datetime):
            start_time = start_time_str
    
    if end_time_str:
        if isinstance(end_time_str, str):
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        elif isinstance(end_time_str, datetime):
            end_time = end_time_str

    # Build agent payload from state
    payload: Dict[str, Any] = {
        "name": agent_state.fields.get("name", ""),
        "camera_id": agent_state.fields.get("camera_id"),
        "model": agent_state.fields.get("model"),
        "fps": agent_state.fields.get("fps"),
        "rules": agent_state.fields.get("rules", []),
        "run_mode": run_mode,
        "start_time": start_time,
        "end_time": end_time,
        "requires_zone": requires_zone,
        "status": "PENDING",
        "created_at": datetime.now(),
        "owner_user_id": user_id,
    }
    
    # Only include interval_minutes and check_duration_seconds for patrol mode
    # DO NOT include these fields at all if run_mode is "continuous" (not even as null)
    if run_mode == "patrol":
        interval_minutes = agent_state.fields.get("interval_minutes")
        check_duration = agent_state.fields.get("check_duration_seconds")
        if interval_minutes is not None:
            payload["interval_minutes"] = interval_minutes
        if check_duration is not None:
            payload["check_duration_seconds"] = check_duration
    
    # Only include zone if requires_zone is true
    # DO NOT include zone field at all if requires_zone is false (not even as null)
    if requires_zone:
        zone = agent_state.fields.get("zone")
        if zone is not None:
            payload["zone"] = zone

    # Create domain model (will validate)
    agent = Agent(**payload)

    # Simple async call - use asyncio.run() directly
    # If nest_asyncio is installed, it will handle nested loops automatically
    # Otherwise, this will work if called from a sync context
    saved_agent = asyncio.run(_agent_repository.save(agent))

    # Register agent with Jetson backend if dependencies are available
    if saved_agent and saved_agent.id and saved_agent.camera_id:
        try:
            asyncio.run(_register_agent_with_jetson(saved_agent))
        except Exception as e:
            # Log error but don't fail the save operation
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Failed to register agent {saved_agent.id} with Jetson backend: {e}",
                exc_info=True
            )

    # Reset for a fresh run after save
    reset_agent_state(session_id)

    return {
        "status": "DONE",
        "saved": True,
        "message": "Agent configuration saved.Tell user agent is successfully created",
        "agent_id": saved_agent.id if saved_agent else None,
    }


async def _convert_agent_to_vision_core_format(
    agent: Agent,
    camera_stream_url: str
) -> Dict[str, Any]:
    """
    Convert vision-backend agent format to vision_core format.
    
    Args:
        agent: Agent domain model from vision-backend
        camera_stream_url: RTSP stream URL from camera
        
    Returns:
        Dictionary in vision_core AgentCreate format
    """
    # Map vision-backend rules to vision_core rules format
    vision_core_rules = []
    for rule in agent.rules:
        if isinstance(rule, dict):
            vision_core_rules.append(rule)
        else:
            vision_core_rules.append(rule.dict() if hasattr(rule, 'dict') else rule)
    
    # Determine task_type from model or rules
    task_type = "object_detection"  # Default
    if agent.rules:
        first_rule = agent.rules[0] if agent.rules else {}
        if isinstance(first_rule, dict):
            rule_type = first_rule.get("type", "")
            if "classification" in rule_type.lower():
                task_type = "classification"
            elif "tracking" in rule_type.lower():
                task_type = "object_tracking"
    
    # Convert status
    status = agent.status.lower() if agent.status else "pending"
    if status == "active":
        status = "running"
    elif status == "created":
        status = "pending"
    
    # Ensure start_time and end_time are datetime objects
    start_at = agent.start_time or datetime.now()
    end_at = agent.end_time or datetime.now()
    
    return {
        "agent_id": agent.id or "",
        "task_name": agent.name,
        "task_type": task_type,
        "camera_id": agent.camera_id,
        "source_uri": camera_stream_url,
        "model_ids": [agent.model] if agent.model else [],
        "fps": agent.fps or 5,
        "run_mode": agent.run_mode or "continuous",
        "rules": vision_core_rules,
        "status": status,
        "start_at": start_at,
        "end_at": end_at,
    }


async def _register_agent_with_jetson(agent: Agent) -> None:
    """
    Register agent with Jetson backend.
    
    This function:
    1. Gets the camera to find device_id and stream_url
    2. Gets the device to find jetson_backend_url
    3. Creates a JetsonClient with the device-specific URL
    4. Converts agent to vision_core format
    5. Sends agent config to Jetson backend
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not _camera_repository or not _jetson_client:
        logger.warning(
            f"Camera repository or JetsonClient not initialized. "
            f"Skipping Jetson registration for agent {agent.id}"
        )
        return  # Dependencies not set, skip registration
    
    # Get camera to find device_id and stream_url
    try:
        camera = await _camera_repository.find_by_id(agent.camera_id)
        if not camera:
            logger.warning(
                f"Camera {agent.camera_id} not found. Skipping Jetson registration for agent {agent.id}"
            )
            return
        
        if not camera.stream_url:
            logger.warning(
                f"Camera {agent.camera_id} has no stream_url. Skipping Jetson registration for agent {agent.id}"
            )
            return
    except Exception as e:
        logger.error(
            f"Error fetching camera {agent.camera_id} for agent {agent.id}: {e}",
            exc_info=True
        )
        return
    
    # Get device backend URL if camera has device_id
    jetson_backend_url = None
    if camera.device_id:
        if not _device_repository:
            logger.warning(
                f"Device repository not initialized. Cannot get Jetson backend URL for device {camera.device_id}. "
                f"Using default JetsonClient for agent {agent.id}"
            )
        else:
            try:
                device = await _device_repository.find_by_id(camera.device_id)
                if device:
                    jetson_backend_url = device.jetson_backend_url
                    logger.info(
                        f"Found device {camera.device_id} with Jetson backend at {jetson_backend_url} "
                        f"for agent {agent.id}"
                    )
                else:
                    logger.warning(
                        f"Device {camera.device_id} not found. Using default JetsonClient for agent {agent.id}"
                    )
            except Exception as e:
                logger.error(
                    f"Error fetching device {camera.device_id} for agent {agent.id}: {e}",
                    exc_info=True
                )
                # Continue with default client
    
    # Create JetsonClient with device-specific URL if available
    from ...infrastructure.external.jetson_client import JetsonClient
    jetson_client = _jetson_client
    if jetson_backend_url:
        jetson_client = JetsonClient(base_url=jetson_backend_url)
        logger.info(
            f"Using device-specific Jetson backend URL: {jetson_backend_url} for agent {agent.id}"
        )
    else:
        logger.info(
            f"Using default Jetson backend URL: {jetson_client.base_url} for agent {agent.id}"
        )
    
    # Convert to vision_core format
    try:
        vision_core_payload = await _convert_agent_to_vision_core_format(
            agent,
            camera.stream_url
        )
    except Exception as e:
        logger.error(
            f"Error converting agent {agent.id} to vision_core format: {e}",
            exc_info=True
        )
        return
    
    # Register agent with Jetson backend
    try:
        logger.info(
            f"Registering agent {agent.id} with Jetson backend at {jetson_client.base_url}"
        )
        success = await jetson_client.register_agent(
            agent_id=vision_core_payload["agent_id"],
            task_name=vision_core_payload["task_name"],
            task_type=vision_core_payload["task_type"],
            camera_id=vision_core_payload["camera_id"],
            source_uri=vision_core_payload["source_uri"],
            model_ids=vision_core_payload["model_ids"],
            fps=vision_core_payload["fps"],
            run_mode=vision_core_payload["run_mode"],
            rules=vision_core_payload["rules"],
            status=vision_core_payload["status"],
            start_at=vision_core_payload["start_at"],
            end_at=vision_core_payload["end_at"],
        )
        
        if success:
            logger.info(
                f"Successfully registered agent {agent.id} with Jetson backend. "
                f"Agent will be processed by runner on next poll."
            )
        else:
            logger.warning(
                f"Failed to register agent {agent.id} with Jetson backend. "
                f"Agent saved locally but will not be processed until registration succeeds."
            )
    except Exception as e:
        logger.error(
            f"Unexpected error registering agent {agent.id} with Jetson backend: {e}",
            exc_info=True
        )
        # Don't raise - agent is saved locally even if Jetson sync fails


async def async_save_to_db(session_id: str = "default", user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Async version of save_to_db for use in async contexts.
    """
    if _agent_repository is None:
        raise ValueError("Agent repository not initialized. Call set_agent_repository() first.")

    agent_state = get_agent_state(session_id)

    if agent_state.status != "CONFIRMATION":
        raise ValueError("Cannot save: agent is not in CONFIRMATION state.")

    if agent_state.missing_fields:
        raise ValueError(f"Cannot save: missing fields {agent_state.missing_fields}")

    # Validate user_id is provided
    if not user_id:
        raise ValueError("Cannot save: user_id is required. Agent must be associated with an authenticated user.")

    # Validate run_mode
    run_mode = agent_state.fields.get("run_mode")
    if run_mode and run_mode not in ["continuous", "patrol"]:
        raise ValueError(f"Invalid run_mode: {run_mode}. Only 'continuous' or 'patrol' are allowed.")
    
    # Load rule to check requires_zone
    from .initialize_state_tool import _get_rule
    rule = _get_rule(agent_state.rule_id) if agent_state.rule_id else None
    requires_zone = bool(rule.get("requires_zone", False)) if rule else agent_state.fields.get("requires_zone", False)
    
    # Convert string times to datetime objects
    start_time_str = agent_state.fields.get("start_time")
    end_time_str = agent_state.fields.get("end_time")
    start_time = None
    end_time = None
    
    if start_time_str:
        if isinstance(start_time_str, str):
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        elif isinstance(start_time_str, datetime):
            start_time = start_time_str
    
    if end_time_str:
        if isinstance(end_time_str, str):
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        elif isinstance(end_time_str, datetime):
            end_time = end_time_str
    
    # Build agent payload from state
    payload: Dict[str, Any] = {
        "name": agent_state.fields.get("name", ""),
        "camera_id": agent_state.fields.get("camera_id"),
        "model": agent_state.fields.get("model"),
        "fps": agent_state.fields.get("fps"),
        "rules": agent_state.fields.get("rules", []),
        "run_mode": run_mode,
        "start_time": start_time,
        "end_time": end_time,
        "requires_zone": requires_zone,
        "status": "PENDING",
        "created_at": datetime.now(),
        "owner_user_id": user_id,
    }
    
    # Only include interval_minutes and check_duration_seconds for patrol mode
    # DO NOT include these fields at all if run_mode is "continuous" (not even as null)
    if run_mode == "patrol":
        interval_minutes = agent_state.fields.get("interval_minutes")
        check_duration = agent_state.fields.get("check_duration_seconds")
        if interval_minutes is not None:
            payload["interval_minutes"] = interval_minutes
        if check_duration is not None:
            payload["check_duration_seconds"] = check_duration
    
    # Only include zone if requires_zone is true
    # DO NOT include zone field at all if requires_zone is false (not even as null)
    if requires_zone:
        zone = agent_state.fields.get("zone")
        if zone is not None:
            payload["zone"] = zone

    agent = Agent(**payload)
    saved_agent = await _agent_repository.save(agent)
    
    # Register agent with Jetson backend if dependencies are available
    if saved_agent and saved_agent.id and saved_agent.camera_id:
        try:
            await _register_agent_with_jetson(saved_agent)
        except Exception as e:
            # Log error but don't fail the save operation
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Failed to register agent {saved_agent.id} with Jetson backend: {e}",
                exc_info=True
            )
    
    reset_agent_state(session_id)

    return {
        "status": "DONE",
        "saved": True,
        "message": "Agent configuration saved.",
        "agent_id": saved_agent.id if saved_agent else None,
    }
