from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId
from bson.errors import InvalidId

from ...domain.models.agent import Agent
from ...domain.constants import AgentFields
from ...infrastructure.external.agent_client import AgentClient
from ...utils.datetime_utils import utc_now
from ...utils.db import get_collection
from ..session_state.agent_state import get_agent_state
from ..exceptions import (
    DatabaseError,
    DatabaseConnectionError,
    ExternalServiceError,
    JetsonRegistrationError,
    RepositoryNotInitializedError,
    ValidationError,
    InvalidStateTransitionError,
    VisionAgentError,
)
from ..utils.retry_utils import retry_on_exception, async_retry_on_exception
from .flow_diagram_utils import generate_agent_flow_diagram

logger = logging.getLogger(__name__)



# ============================================================================
# REPOSITORY MANAGEMENT
# ============================================================================

_agent_repository: Optional[Any] = None
_camera_repository: Optional[Any] = None
_device_repository: Optional[Any] = None
_jetson_client: Optional[Any] = None


def set_agent_repository(repository):
    """Set the agent repository for saving agents."""
    global _agent_repository
    _agent_repository = repository


def set_camera_repository(repository):
    """Set the camera repository for looking up cameras."""
    global _camera_repository
    _camera_repository = repository


def get_camera_repository():
    """Get the camera repository (for use by other tools)."""
    return _camera_repository


# ============================================================================
# SYNC AGENT SAVE (no asyncio – works in Docker and from running event loop)
# ============================================================================

def _agent_to_dict_sync(agent: Agent) -> dict:
    """Convert Agent to MongoDB document (sync, same shape as MongoAgentRepository)._agent_to_dict)."""
    d = {
        AgentFields.NAME: agent.name,
        AgentFields.CAMERA_ID: agent.camera_id,
        AgentFields.MODEL: agent.model,
        AgentFields.FPS: agent.fps,
        AgentFields.RULES: agent.rules,
        AgentFields.RUN_MODE: agent.run_mode,
        AgentFields.INTERVAL_MINUTES: agent.interval_minutes,
        AgentFields.CHECK_DURATION_SECONDS: agent.check_duration_seconds,
        AgentFields.START_TIME: agent.start_time,
        AgentFields.END_TIME: agent.end_time,
        AgentFields.ZONE: agent.zone,
        AgentFields.REQUIRES_ZONE: agent.requires_zone,
        AgentFields.STATUS: agent.status,
        AgentFields.CREATED_AT: agent.created_at,
        AgentFields.OWNER_USER_ID: agent.owner_user_id,
        AgentFields.STREAM_CONFIG: agent.stream_config,
    }
    if agent.id:
        try:
            d[AgentFields.MONGO_ID] = ObjectId(agent.id)
        except (InvalidId, ValueError, TypeError):
            pass
    return d


def _document_to_agent_sync(document: dict) -> Agent:
    """Convert MongoDB document to Agent (sync, same shape as MongoAgentRepository._document_to_agent)."""
    if not document or AgentFields.MONGO_ID not in document:
        raise ValueError("Invalid document: missing _id field")
    start_time = document.get(AgentFields.START_TIME)
    end_time = document.get(AgentFields.END_TIME)
    if start_time and isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if end_time and isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    return Agent(
        id=str(document[AgentFields.MONGO_ID]),
        name=document.get(AgentFields.NAME, ""),
        camera_id=document.get(AgentFields.CAMERA_ID, ""),
        model=document.get(AgentFields.MODEL, ""),
        fps=document.get(AgentFields.FPS),
        rules=document.get(AgentFields.RULES, []),
        run_mode=document.get(AgentFields.RUN_MODE),
        interval_minutes=document.get(AgentFields.INTERVAL_MINUTES),
        check_duration_seconds=document.get(AgentFields.CHECK_DURATION_SECONDS),
        start_time=start_time,
        end_time=end_time,
        zone=document.get(AgentFields.ZONE),
        requires_zone=document.get(AgentFields.REQUIRES_ZONE, False),
        status=document.get(AgentFields.STATUS, "ACTIVE"),
        created_at=document.get(AgentFields.CREATED_AT),
        owner_user_id=document.get(AgentFields.OWNER_USER_ID),
        stream_config=document.get(AgentFields.STREAM_CONFIG),
    )


@retry_on_exception(max_retries=3, initial_delay=1.0)
def _save_agent_sync(agent: Agent) -> Agent:
    """
    Save agent using sync PyMongo with retry logic.
    
    Args:
        agent: Agent instance to save
        
    Returns:
        Agent: Saved agent with ID
        
    Raises:
        DatabaseError: If save operation fails
        DatabaseConnectionError: If database connection fails
    """
    try:
        logger.info(f"Saving agent: {agent.name} (camera_id={agent.camera_id})")
        coll = get_collection("agents")
        agent_dict = _agent_to_dict_sync(agent)
        
        # Update existing agent
        if agent.id:
            try:
                oid = ObjectId(agent.id)
                logger.debug(f"Updating existing agent with ID: {agent.id}")
                
                result = coll.update_one(
                    {AgentFields.MONGO_ID: oid},
                    {"$set": {k: v for k, v in agent_dict.items() if k != AgentFields.MONGO_ID}},
                )
                
                if result.matched_count == 0:
                    logger.warning(f"Agent {agent.id} not found for update, will insert instead")
                    # Agent not found, fall through to insert
                else:
                    doc = coll.find_one({AgentFields.MONGO_ID: oid})
                    if doc is None:
                        raise DatabaseError(
                            f"Agent {agent.id} was updated but could not be retrieved",
                            operation="update",
                            retryable=False
                        )
                    logger.info(f"Agent updated successfully: {agent.id}")
                    return _document_to_agent_sync(doc)
                    
            except InvalidId as e:
                logger.warning(f"Invalid agent ID format: {agent.id}: {e}")
                # Continue to insert
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid agent ID type: {agent.id}: {e}")
                # Continue to insert
        
        # Insert new agent
        if AgentFields.MONGO_ID in agent_dict:
            del agent_dict[AgentFields.MONGO_ID]
        
        logger.debug("Inserting new agent")
        result = coll.insert_one(agent_dict)
        
        doc = coll.find_one({AgentFields.MONGO_ID: result.inserted_id})
        if doc is None:
            raise DatabaseError(
                "Agent was created but could not be retrieved",
                operation="insert",
                retryable=False
            )
        
        saved_agent = _document_to_agent_sync(doc)
        logger.info(f"Agent created successfully: {saved_agent.id}")
        return saved_agent
        
    except DatabaseError:
        # Re-raise DatabaseError
        raise
    except Exception as e:
        logger.exception(f"Failed to save agent: {e}")
        raise DatabaseError(
            f"Failed to save agent to database: {str(e)}",
            operation="save",
            retryable=True,
            user_message="Failed to save agent. Please try again."
        )


def set_device_repository(repository):
    """Set the device repository for looking up devices."""
    global _device_repository
    _device_repository = repository


def set_jetson_client(client):
    """Set the Jetson client for registering agents."""
    global _jetson_client
    _jetson_client = client


# ============================================================================
# AGENT SAVING
# ============================================================================


async def save_to_db_async(
    session_id: str = "default", user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Non-blocking save. Runs sync save_to_db in thread pool.
    Used by Agent Creation chat to avoid blocking the event loop.
    
    Args:
        session_id: Session identifier
        user_id: User ID who owns the agent
        
    Returns:
        Dict with save result
    """
    try:
        logger.debug(f"save_to_db_async: session={session_id}, user_id={user_id}")
        result = await asyncio.to_thread(
            save_to_db, session_id=session_id, user_id=user_id
        )
        logger.debug(f"save_to_db_async completed: saved={result.get('saved')}")
        return result
    except VisionAgentError as e:
        # Known errors - return structured response
        logger.error(f"save_to_db_async failed with VisionAgentError: {e}")
        return {
            "error": e.message,
            "user_message": e.user_message,
            "status": "COLLECTING",
            "saved": False,
            "message": e.user_message,
        }
    except Exception as e:
        # Unexpected errors - log and return generic response
        logger.exception(f"Unexpected error in save_to_db_async: {e}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "status": "COLLECTING",
            "saved": False,
            "message": "An unexpected error occurred. Please try again.",
        }


def save_to_db(session_id: str = "default", user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Finalize the agent configuration and save to database.

    Args:
        session_id: Session identifier for state management
        user_id: User ID who owns this agent

    Returns:
        Dict with status, saved flag, agent_id, and agent_name
        
    Raises:
        RepositoryNotInitializedError: If repository is not initialized
        ValidationError: If validation fails
        DatabaseError: If database operation fails
    """
    try:
        logger.info(f"save_to_db called: session={session_id}, user_id={user_id}")
        
        if _agent_repository is None:
            error_msg = "Agent repository not initialized. Call set_agent_repository() first."
            logger.error(error_msg)
            raise RepositoryNotInitializedError("AgentRepository")

        agent_state = get_agent_state(session_id)
        logger.debug(f"Agent state: status={agent_state.status}, rule_id={agent_state.rule_id}")

        # Auto-transition from COLLECTING to CONFIRMATION if ready
        if agent_state.status == "COLLECTING" and not agent_state.missing_fields:
            from .kb_utils import compute_missing_fields, get_rule
            try:
                rule = get_rule(agent_state.rule_id) if agent_state.rule_id else None
                if rule:
                    compute_missing_fields(agent_state, rule)
                    if not agent_state.missing_fields:
                        agent_state.status = "CONFIRMATION"
                        logger.info(f"Auto-transitioned session {session_id} to CONFIRMATION state")
            except Exception as e:
                logger.warning(f"Failed to compute missing fields during auto-transition: {e}")

        # Validate state is ready for saving
        if agent_state.status != "CONFIRMATION":
            error_msg = f"Cannot save: agent is not in CONFIRMATION state. Current status: {agent_state.status}"
            logger.warning(error_msg)
            raise InvalidStateTransitionError(
                error_msg,
                user_message="Agent is not ready to save. Please complete all required fields first."
            )

        if agent_state.missing_fields:
            error_msg = f"Cannot save: missing fields {agent_state.missing_fields}"
            logger.warning(f"{error_msg} for session {session_id}")
            raise ValidationError(
                error_msg,
                user_message=f"Please provide the following fields: {', '.join(agent_state.missing_fields)}"
            )

        if not user_id:
            error_msg = "Cannot save: user_id is required. Agent must be associated with an authenticated user."
            logger.error(error_msg)
            raise ValidationError(
                error_msg,
                user_message="Authentication error: User ID is required."
            )

        # Validate run_mode
        run_mode = agent_state.fields.get("run_mode")
        if run_mode and run_mode not in ["continuous", "patrol"]:
            error_msg = f"Invalid run_mode: {run_mode}. Only 'continuous' or 'patrol' are allowed."
            logger.error(error_msg)
            raise ValidationError(
                error_msg,
                user_message=f"Invalid run mode: {run_mode}. Please use 'continuous' or 'patrol'."
            )

        # Get rule and compute zone requirement
        from .kb_utils import compute_requires_zone, get_rule
        try:
            rule = get_rule(agent_state.rule_id) if agent_state.rule_id else None
            logger.debug(f"Retrieved rule: {agent_state.rule_id}")
        except ValueError as e:
            logger.error(f"Invalid rule_id {agent_state.rule_id}: {e}")
            raise ValidationError(
                f"Invalid rule_id: {str(e)}",
                user_message=f"Configuration error: {str(e)}"
            )

        if rule:
            requires_zone = compute_requires_zone(rule, run_mode)
        else:
            requires_zone = False

        # Parse and validate time fields
        start_time_str = agent_state.fields.get("start_time")
        end_time_str = agent_state.fields.get("end_time")
        start_time = None
        end_time = None

        try:
            if start_time_str:
                if isinstance(start_time_str, str):
                    start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                    logger.debug(f"Parsed start_time: {start_time}")
                elif isinstance(start_time_str, datetime):
                    start_time = start_time_str

            if end_time_str:
                if isinstance(end_time_str, str):
                    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                    logger.debug(f"Parsed end_time: {end_time}")
                elif isinstance(end_time_str, datetime):
                    end_time = end_time_str
            
            # Validate time window if both are provided
            if start_time and end_time and start_time >= end_time:
                raise ValidationError(
                    f"start_time ({start_time}) must be before end_time ({end_time})",
                    user_message="Start time must be before end time."
                )
                
        except ValidationError:
            raise
        except ValueError as e:
            logger.error(f"Invalid time format: {e}")
            raise ValidationError(
                f"Invalid time format: {str(e)}",
                user_message=f"Time format error: {str(e)}"
            )

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
            "created_at": utc_now(),
            "owner_user_id": user_id,
        }

        if run_mode == "patrol":
            interval_minutes = agent_state.fields.get("interval_minutes")
            check_duration = agent_state.fields.get("check_duration_seconds")
            if interval_minutes is not None:
                payload["interval_minutes"] = interval_minutes
            if check_duration is not None:
                payload["check_duration_seconds"] = check_duration

        if requires_zone:
            zone = agent_state.fields.get("zone")
            if zone is not None:
                payload["zone"] = zone

        # Create agent instance
        try:
            agent = Agent(**payload)
            logger.info(f"Created agent instance: {agent.name}")
        except Exception as e:
            logger.exception(f"Invalid agent configuration: {e}")
            raise ValidationError(
                f"Invalid agent configuration: {str(e)}",
                user_message=f"Configuration error: {str(e)}"
            )

        # Save agent to database
        try:
            saved_agent = _save_agent_sync(agent)

            if saved_agent is None:
                error_msg = "Failed to save agent: Repository returned None"
                logger.error(error_msg)
                raise DatabaseError(
                    error_msg,
                    operation="save",
                    retryable=True,
                    user_message="Database error: Agent was not saved. Please try again."
                )

            if not hasattr(saved_agent, 'id') or saved_agent.id is None:
                error_msg = "Failed to save agent: Agent has no ID"
                logger.error(error_msg)
                raise DatabaseError(
                    error_msg,
                    operation="save",
                    retryable=False,
                    user_message="Database error: Agent was not properly saved. Please try again."
                )
            
            logger.info(f"Agent saved successfully: {saved_agent.id}")

        except DatabaseError:
            # Re-raise DatabaseError
            raise
        except Exception as e:
            logger.exception(f"Unexpected error saving agent: {e}")
            raise DatabaseError(
                f"Failed to save agent to database: {str(e)}",
                operation="save",
                retryable=True,
                user_message=f"Database error: {str(e)}"
            )

        # Update state
        agent_state.status = "SAVED"
        agent_state.saved_agent_id = saved_agent.id if saved_agent else None
        agent_state.saved_agent_name = saved_agent.name if saved_agent else None
        logger.info(f"Updated session {session_id} status to SAVED")

        # Generate flow diagram (non-critical - don't fail if this fails)
        flow_diagram = None
        try:
            flow_diagram = generate_agent_flow_diagram(saved_agent)
            logger.debug(f"Generated flow diagram for agent {saved_agent.id}")
        except Exception as e:
            logger.warning(f"Failed to generate flow diagram for agent {saved_agent.id}: {e}")
            # Continue without flow diagram

        result = {
            "status": "SAVED",
            "saved": True,
            "message": "Agent configuration saved. Tell user agent is successfully created",
            "agent_id": saved_agent.id if saved_agent else None,
            "agent_name": saved_agent.name if saved_agent else None,
            "flow_diagram": flow_diagram,
        }
        logger.info(f"save_to_db completed successfully for session {session_id}")
        return result
        
    except (ValidationError, InvalidStateTransitionError, RepositoryNotInitializedError) as e:
        # Expected errors - return structured response
        logger.error(f"Save failed with validation/state error: {e}")
        return {
            "error": e.message,
            "user_message": e.user_message,
            "status": agent_state.status if 'agent_state' in locals() else "COLLECTING",
            "saved": False,
            "message": e.user_message
        }
    except DatabaseError as e:
        # Database errors - return structured response
        logger.error(f"Save failed with database error: {e}")
        return {
            "error": e.message,
            "user_message": e.user_message,
            "status": agent_state.status if 'agent_state' in locals() else "COLLECTING",
            "saved": False,
            "message": e.user_message,
            "retryable": e.retryable
        }
    except Exception as e:
        # Unexpected errors - log and return generic response
        logger.exception(f"Unexpected error saving agent for session {session_id}: {e}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "status": agent_state.status if 'agent_state' in locals() else "COLLECTING",
            "saved": False,
            "message": "An unexpected error occurred. Please try again."
        }


# ============================================================================
# JETSON REGISTRATION
# ============================================================================

async def _register_agent_with_jetson(agent: Agent) -> None:
    """
    Register agent with Jetson backend.

    This function:
    1. Gets the camera to find device_id
    2. Gets the device to find jetson_backend_url
    3. Creates a JetsonClient with the device-specific URL
    4. Sends agent config to Jetson backend in the same format as stored in database
    5. Gets and stores stream config for the agent
    
    Note: This is a non-critical operation. Failures are logged but don't prevent agent creation.
    
    Raises:
        JetsonRegistrationError: If registration fails (logged, not raised to caller)
    """
    if not _jetson_client:
        logger.warning("Jetson client not initialized, skipping agent registration")
        return

    try:
        # Get Jetson backend URL from camera/device
        jetson_backend_url = None
        if _camera_repository:
            try:
                camera = await _camera_repository.find_by_id(agent.camera_id)
                if camera and camera.device_id and _device_repository:
                    try:
                        device = await _device_repository.find_by_id(camera.device_id)
                        if device:
                            jetson_backend_url = device.jetson_backend_url
                            logger.debug(f"Found Jetson backend URL for agent {agent.id}: {jetson_backend_url}")
                    except Exception as e:
                        logger.warning(f"Failed to get device for camera {agent.camera_id}: {e}")
            except Exception as e:
                logger.warning(f"Failed to get camera {agent.camera_id}: {e}")

        jetson_client = _jetson_client
        if jetson_backend_url:
            jetson_client = AgentClient(base_url=jetson_backend_url)

        # Prepare agent dict for registration
        try:
            agent_dict = asdict(agent)
            agent_dict.pop("stream_config", None)
            
            # Convert datetime objects to ISO strings
            if agent_dict.get("start_time") and isinstance(agent_dict["start_time"], datetime):
                agent_dict["start_time"] = agent_dict["start_time"].isoformat()
            if agent_dict.get("end_time") and isinstance(agent_dict["end_time"], datetime):
                agent_dict["end_time"] = agent_dict["end_time"].isoformat()
            if agent_dict.get("created_at") and isinstance(agent_dict["created_at"], datetime):
                agent_dict["created_at"] = agent_dict["created_at"].isoformat()
        except Exception as e:
            logger.error(f"Failed to prepare agent dict for Jetson registration: {e}")
            raise JetsonRegistrationError(f"Failed to prepare agent data: {str(e)}")

        # Register agent with Jetson
        try:
            logger.info(f"Registering agent {agent.id} with Jetson backend")
            success = await jetson_client.register_agent_raw(agent_dict)

            if success:
                logger.info(f"Agent {agent.id} registered successfully with Jetson")
                
                # Get stream config
                try:
                    config_dict = await jetson_client.get_stream_config_for_agent(
                        agent_id=agent.id,
                        camera_id=agent.camera_id,
                        user_id=agent.owner_user_id or ""
                    )

                    if config_dict and _agent_repository:
                        agent.stream_config = config_dict
                        await _agent_repository.save(agent)
                        logger.info(f"Updated stream config for agent {agent.id}")
                except Exception as e:
                    logger.warning(f"Failed to get/save stream config for agent {agent.id}: {e}")
                    # Non-critical - continue
            else:
                logger.error(f"Jetson registration returned failure for agent {agent.id}")
                raise JetsonRegistrationError(f"Registration returned failure for agent {agent.id}")
                
        except JetsonRegistrationError:
            raise
        except Exception as e:
            logger.exception(f"Failed to register agent {agent.id} with Jetson: {e}")
            raise JetsonRegistrationError(f"Registration failed: {str(e)}")
            
    except JetsonRegistrationError as e:
        # Log but don't propagate - agent was already saved to DB
        logger.error(f"Jetson registration failed for agent {agent.id}: {e.message}")
        # Could add to a retry queue here in production
    except Exception as e:
        # Unexpected error - log but don't propagate
        logger.exception(f"Unexpected error during Jetson registration for agent {agent.id}: {e}")


# ============================================================================
# ASYNC VERSION
# ============================================================================

async def async_save_to_db(session_id: str = "default", user_id: Optional[str] = None) -> Dict[str, Any]:
    """Async version of save_to_db for use in async contexts."""
    if _agent_repository is None:
        raise ValueError("Agent repository not initialized. Call set_agent_repository() first.")

    agent_state = get_agent_state(session_id)

    if agent_state.status != "CONFIRMATION":
        raise ValueError("Cannot save: agent is not in CONFIRMATION state.")

    if agent_state.missing_fields:
        raise ValueError(f"Cannot save: missing fields {agent_state.missing_fields}")

    if not user_id:
        raise ValueError("Cannot save: user_id is required. Agent must be associated with an authenticated user.")

    run_mode = agent_state.fields.get("run_mode")
    if run_mode and run_mode not in ["continuous", "patrol"]:
        raise ValueError(f"Invalid run_mode: {run_mode}. Only 'continuous' or 'patrol' are allowed.")

    from .kb_utils import compute_requires_zone, get_rule
    rule = get_rule(agent_state.rule_id) if agent_state.rule_id else None

    if rule:
        requires_zone = compute_requires_zone(rule, run_mode)
    else:
        requires_zone = False

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

    if run_mode == "patrol":
        interval_minutes = agent_state.fields.get("interval_minutes")
        check_duration = agent_state.fields.get("check_duration_seconds")
        if interval_minutes is not None:
            payload["interval_minutes"] = interval_minutes
        if check_duration is not None:
            payload["check_duration_seconds"] = check_duration

    if requires_zone:
        zone = agent_state.fields.get("zone")
        if zone is not None:
            payload["zone"] = zone

    agent = Agent(**payload)
    saved_agent = await _agent_repository.save(agent)

    if saved_agent and saved_agent.id and saved_agent.camera_id:
        try:
            await _register_agent_with_jetson(saved_agent)
        except Exception:
            pass

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
