from __future__ import annotations

from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from dataclasses import asdict

from ..session_state.agent_state import get_agent_state, reset_agent_state
from ...domain.models.agent import Agent
from ...infrastructure.external.agent_client import AgentClient
from .flow_diagram_utils import generate_agent_flow_diagram

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
    print(f"[set_agent_repository] Agent repository set: {type(repository)}")


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
    print(f"[save_to_db] Starting save operation for session_id={session_id}, user_id={user_id}")
    try:
        if _agent_repository is None:
            print("[save_to_db] ERROR: Agent repository not initialized")
            return {
                "error": "Agent repository not initialized. Call set_agent_repository() first.",
                "status": "COLLECTING",
                "saved": False,
                "message": "System error: Agent repository not available."
            }

        print(f"[save_to_db] Agent repository is initialized: {type(_agent_repository)}")
        agent_state = get_agent_state(session_id)
        print(f"[save_to_db] Agent state retrieved - status: {agent_state.status}, rule_id: {agent_state.rule_id}, missing_fields: {agent_state.missing_fields}")
        
        # Safety check: If missing_fields is empty but status is still COLLECTING, 
        # recompute missing fields and transition to CONFIRMATION if ready
        if agent_state.status == "COLLECTING" and not agent_state.missing_fields:
            print(f"[save_to_db] WARNING: Status is COLLECTING but no missing fields. Recomputing missing fields...")
            from .kb_utils import get_rule, compute_missing_fields
            try:
                rule = get_rule(agent_state.rule_id) if agent_state.rule_id else None
                if rule:
                    compute_missing_fields(agent_state, rule)
                    print(f"[save_to_db] After recompute - missing_fields: {agent_state.missing_fields}")
                    if not agent_state.missing_fields:
                        agent_state.status = "CONFIRMATION"
                        print(f"[save_to_db] Status auto-transitioned to CONFIRMATION")
            except Exception as e:
                print(f"[save_to_db] Error during recompute: {e}")
        
        if agent_state.status != "CONFIRMATION":
            print(f"[save_to_db] ERROR: Agent not in CONFIRMATION state. Current status: {agent_state.status}, missing_fields: {agent_state.missing_fields}")
            return {
                "error": f"Cannot save: agent is not in CONFIRMATION state. Current status: {agent_state.status}",
                "status": agent_state.status,
                "saved": False,
                "message": "Agent is not ready to save. Please complete all required fields first."
            }

        if agent_state.missing_fields:
            print(f"[save_to_db] ERROR: Missing fields: {agent_state.missing_fields}")
            return {
                "error": f"Cannot save: missing fields {agent_state.missing_fields}",
                "status": agent_state.status,
                "saved": False,
                "message": f"Please provide the following fields: {', '.join(agent_state.missing_fields)}"
            }

        # Validate user_id is provided
        if not user_id:
            print("[save_to_db] ERROR: user_id is not provided")
            return {
                "error": "Cannot save: user_id is required. Agent must be associated with an authenticated user.",
                "status": agent_state.status,
                "saved": False,
                "message": "Authentication error: User ID is required."
            }
        
        print(f"[save_to_db] Validation passed - proceeding to create agent model")

        # Validate run_mode
        run_mode = agent_state.fields.get("run_mode")
        if run_mode and run_mode not in ["continuous", "patrol"]:
            return {
                "error": f"Invalid run_mode: {run_mode}. Only 'continuous' or 'patrol' are allowed.",
                "status": agent_state.status,
                "saved": False,
                "message": f"Invalid run mode: {run_mode}. Please use 'continuous' or 'patrol'."
            }
        
        # Load rule to compute requires_zone (derived state, not stored)
        from .kb_utils import get_rule, compute_requires_zone
        try:
            rule = get_rule(agent_state.rule_id) if agent_state.rule_id else None
        except ValueError as e:
            return {
                "error": f"Invalid rule_id: {str(e)}",
                "status": agent_state.status,
                "saved": False,
                "message": f"Configuration error: {str(e)}"
            }
        
        # Compute requires_zone from KB structure (derived, not stored)
        if rule:
            requires_zone = compute_requires_zone(rule, run_mode)
        else:
            # Fallback if rule not found (shouldn't happen, but be safe)
            requires_zone = False

        # Convert string times to datetime objects
        start_time_str = agent_state.fields.get("start_time")
        end_time_str = agent_state.fields.get("end_time")
        start_time = None
        end_time = None
        
        try:
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
        except ValueError as e:
            return {
                "error": f"Invalid time format: {str(e)}",
                "status": agent_state.status,
                "saved": False,
                "message": f"Time format error: {str(e)}"
            }

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
        try:
            print(f"[save_to_db] Creating Agent domain model with payload keys: {list(payload.keys())}")
            agent = Agent(**payload)
            print(f"[save_to_db] Agent domain model created successfully")
        except Exception as e:
            print(f"[save_to_db] ERROR: Failed to create Agent domain model: {str(e)}")
            import traceback
            print(f"[save_to_db] Traceback: {traceback.format_exc()}")
            return {
                "error": f"Invalid agent configuration: {str(e)}",
                "status": agent_state.status,
                "saved": False,
                "message": f"Configuration error: {str(e)}"
            }

        # Simple async call - use asyncio.run() directly
        # If nest_asyncio is installed, it will handle nested loops automatically
        # Otherwise, this will work if called from a sync context
        try:
            print(f"[save_to_db] Attempting to save agent to database...")
            saved_agent = asyncio.run(_agent_repository.save(agent))
            print(f"[save_to_db] Repository save() returned: {saved_agent}")
            print(f"[save_to_db] Saved agent ID: {saved_agent.id if saved_agent else 'None'}")
            print(f"[save_to_db] Saved agent name: {saved_agent.name if saved_agent else 'None'}")
            
            # Validate that save actually worked
            if saved_agent is None:
                print("[save_to_db] ERROR: Repository save() returned None - agent was not saved")
                return {
                    "error": "Failed to save agent: Repository returned None",
                    "status": agent_state.status,
                    "saved": False,
                    "message": "Database error: Agent was not saved. Please try again."
                }
            
            if not hasattr(saved_agent, 'id') or saved_agent.id is None:
                print("[save_to_db] ERROR: Saved agent has no ID - agent was not properly saved")
                return {
                    "error": "Failed to save agent: Agent has no ID",
                    "status": agent_state.status,
                    "saved": False,
                    "message": "Database error: Agent was not properly saved. Please try again."
                }
            
            print(f"[save_to_db] Agent successfully saved with ID: {saved_agent.id}")
        except Exception as e:
            print(f"[save_to_db] ERROR: Exception during repository save: {str(e)}")
            import traceback
            print(f"[save_to_db] Traceback: {traceback.format_exc()}")
            return {
                "error": f"Failed to save agent to database: {str(e)}",
                "status": agent_state.status,
                "saved": False,
                "message": f"Database error: {str(e)}"
            }

        # Register agent with Jetson backend if dependencies are available
        if saved_agent and saved_agent.id and saved_agent.camera_id:
            try:
                print(f"[save_to_db] Registering agent {saved_agent.id} with Jetson backend...")
                asyncio.run(_register_agent_with_jetson(saved_agent))
                print(f"[save_to_db] Successfully registered agent with Jetson backend")
            except Exception as e:
                # Print error but don't fail the save operation
                print(f"[save_to_db] WARNING: Failed to register agent {saved_agent.id} with Jetson backend: {e}")
                import traceback
                print(f"[save_to_db] Traceback: {traceback.format_exc()}")

        # Update state to SAVED instead of resetting
        # DO NOT clear rule_id, fields, or missing_fields - they are harmless once status is SAVED
        # SAVED is a frozen, read-only terminal state
        agent_state.status = "SAVED"
        agent_state.saved_agent_id = saved_agent.id if saved_agent else None
        agent_state.saved_agent_name = saved_agent.name if saved_agent else None
        
        print(f"[save_to_db] Agent state updated to SAVED with agent_id={saved_agent.id}, agent_name={saved_agent.name}")
        
        # Generate flow diagram data for the saved agent
        flow_diagram = None
        try:
            flow_diagram = generate_agent_flow_diagram(saved_agent)
            print(f"[save_to_db] Generated flow diagram with {len(flow_diagram.get('nodes', []))} nodes and {len(flow_diagram.get('links', []))} links")
        except Exception as e:
            print(f"[save_to_db] WARNING: Failed to generate flow diagram: {e}")
            import traceback
            print(f"[save_to_db] Traceback: {traceback.format_exc()}")
        
        result = {
            "status": "SAVED",
            "saved": True,
            "message": "Agent configuration saved. Tell user agent is successfully created",
            "agent_id": saved_agent.id if saved_agent else None,
            "agent_name": saved_agent.name if saved_agent else None,
            "flow_diagram": flow_diagram,  # Include Sankey diagram data
        }
        print(f"[save_to_db] Returning success result: {result}")
        return result
    except Exception as e:
        print(f"[save_to_db] ERROR: Unexpected exception in save_to_db: {str(e)}")
        import traceback
        print(f"[save_to_db] Traceback: {traceback.format_exc()}")
        return {
            "error": f"Unexpected error saving agent: {str(e)}",
            "status": "COLLECTING",
            "saved": False,
            "message": f"An unexpected error occurred: {str(e)}"
        }


async def _register_agent_with_jetson(agent: Agent) -> None:
    """
    Register agent with Jetson backend.
    
    This function:
    1. Gets the camera to find device_id
    2. Gets the device to find jetson_backend_url
    3. Creates a JetsonClient with the device-specific URL
    4. Sends agent config to Jetson backend in the same format as stored in database
    5. Gets and stores stream config for the agent
    """
    print(f"[_register_agent_with_jetson] Starting registration for agent {agent.id}")
    
    if not _jetson_client:
        print(
            f"[_register_agent_with_jetson] WARNING: JetsonClient not initialized. "
            f"Skipping Jetson registration for agent {agent.id}"
        )
        return  # Dependencies not set, skip registration
    
    # Get camera to find device_id
    jetson_backend_url = None
    if _camera_repository:
        try:
            camera = await _camera_repository.find_by_id(agent.camera_id)
            if camera and camera.device_id and _device_repository:
                try:
                    device = await _device_repository.find_by_id(camera.device_id)
                    if device:
                        jetson_backend_url = device.jetson_backend_url
                        print(
                            f"[_register_agent_with_jetson] Found device {camera.device_id} with Jetson backend at {jetson_backend_url} "
                            f"for agent {agent.id}"
                        )
                except Exception as e:
                    print(
                        f"[_register_agent_with_jetson] ERROR: Error fetching device {camera.device_id} for agent {agent.id}: {e}"
                    )
                    import traceback
                    print(f"[_register_agent_with_jetson] Traceback: {traceback.format_exc()}")
        except Exception as e:
            print(
                f"[_register_agent_with_jetson] ERROR: Error fetching camera {agent.camera_id} for agent {agent.id}: {e}"
            )
            import traceback
            print(f"[_register_agent_with_jetson] Traceback: {traceback.format_exc()}")
    
    # Create AgentClient with device-specific URL if available
    jetson_client = _jetson_client
    if jetson_backend_url:
        jetson_client = AgentClient(base_url=jetson_backend_url)
        print(
            f"[_register_agent_with_jetson] Using device-specific Jetson backend URL: {jetson_backend_url} for agent {agent.id}"
        )
    else:
        print(
            f"[_register_agent_with_jetson] Using default Jetson backend URL: {jetson_client.base_url} for agent {agent.id}"
        )
    
    # Convert agent to dict (same format as stored in database)
    try:
        agent_dict = asdict(agent)
        # Remove stream_config from payload - we'll get it from Jetson after registration
        agent_dict.pop("stream_config", None)
        # Convert datetime objects to ISO format strings for JSON serialization
        if agent_dict.get("start_time") and isinstance(agent_dict["start_time"], datetime):
            agent_dict["start_time"] = agent_dict["start_time"].isoformat()
        if agent_dict.get("end_time") and isinstance(agent_dict["end_time"], datetime):
            agent_dict["end_time"] = agent_dict["end_time"].isoformat()
        if agent_dict.get("created_at") and isinstance(agent_dict["created_at"], datetime):
            agent_dict["created_at"] = agent_dict["created_at"].isoformat()
        
        # Rules are already in correct format (no conversion needed)
        # Keep rules as-is to maintain same structure
    except Exception as e:
        print(
            f"[_register_agent_with_jetson] ERROR: Error converting agent {agent.id} to dict: {e}"
        )
        import traceback
        print(f"[_register_agent_with_jetson] Traceback: {traceback.format_exc()}")
        return
    
    # Register agent with Jetson backend using raw agent config
    try:
        print(
            f"[_register_agent_with_jetson] Registering agent {agent.id} with Jetson backend at {jetson_client.base_url}"
        )
        success = await jetson_client.register_agent_raw(agent_dict)
        
        if success:
            print(
                f"[_register_agent_with_jetson] Successfully registered agent {agent.id} with Jetson backend. "
                f"Fetching stream config..."
            )
            
            # Get stream config for this specific agent
            config_dict = await jetson_client.get_stream_config_for_agent(
                agent_id=agent.id,
                camera_id=agent.camera_id,
                user_id=agent.owner_user_id or ""
            )
            
            if config_dict and _agent_repository:
                # Store stream config in agent
                agent.stream_config = config_dict
                updated_agent = await _agent_repository.save(agent)
                print(
                    f"[_register_agent_with_jetson] Successfully registered agent {agent.id} with Jetson backend "
                    f"and stored stream config"
                )
            else:
                print(
                    f"[_register_agent_with_jetson] WARNING: Agent {agent.id} registered with Jetson backend but failed to get stream config. "
                    f"Agent will work but stream viewing may not be available."
                )
        else:
            print(
                f"[_register_agent_with_jetson] WARNING: Failed to register agent {agent.id} with Jetson backend. "
                f"Agent saved locally but will not be processed until registration succeeds."
            )
    except Exception as e:
        print(
            f"[_register_agent_with_jetson] ERROR: Unexpected error registering agent {agent.id} with Jetson backend: {e}"
        )
        import traceback
        print(f"[_register_agent_with_jetson] Traceback: {traceback.format_exc()}")
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
    
    # Load rule to compute requires_zone (derived state, not stored)
    from .kb_utils import get_rule, compute_requires_zone
    rule = get_rule(agent_state.rule_id) if agent_state.rule_id else None
    
    # Compute requires_zone from KB structure (derived, not stored)
    if rule:
        requires_zone = compute_requires_zone(rule, run_mode)
    else:
        # Fallback if rule not found (shouldn't happen, but be safe)
        requires_zone = False
    
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
            # Print error but don't fail the save operation
            print(f"[async_save_to_db] WARNING: Failed to register agent {saved_agent.id} with Jetson backend: {e}")
            import traceback
            print(f"[async_save_to_db] Traceback: {traceback.format_exc()}")
    
    # Update state to SAVED instead of resetting
    # DO NOT clear rule_id, fields, or missing_fields - they are harmless once status is SAVED
    # SAVED is a frozen, read-only terminal state
    agent_state.status = "SAVED"
    agent_state.saved_agent_id = saved_agent.id if saved_agent else None
    agent_state.saved_agent_name = saved_agent.name if saved_agent else None

    return {
        "status": "SAVED",
        "saved": True,
        "message": "Agent configuration saved.",
        "agent_id": saved_agent.id if saved_agent else None,
        "agent_name": saved_agent.name if saved_agent else None,
    }
