from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId
from bson.errors import InvalidId

from ...domain.models.agent import Agent
from ...domain.constants import AgentFields
from ...utils.datetime_utils import utc_now
from ...utils.db import get_collection
from ..session_state.agent_state import get_agent_state
from ..exceptions import (
    DatabaseError,
    DatabaseConnectionError,
    ExternalServiceError,
    RepositoryNotInitializedError,
    ValidationError,
    InvalidStateTransitionError,
    VisionAgentError,
)
from .flow_diagram_utils import generate_agent_flow_diagram

logger = logging.getLogger(__name__)



# ============================================================================
# REPOSITORY MANAGEMENT
# ============================================================================

_agent_repository: Optional[Any] = None
_camera_repository: Optional[Any] = None


def set_agent_repository(repository):
    """Set the agent repository for saving agents."""
    global _agent_repository
    _agent_repository = repository
    print(f"[set_agent_repository] Agent repository set: {type(repository)}")


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
        AgentFields.CAMERA_ID: agent.camera_id or "",
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
        AgentFields.VIDEO_PATH: getattr(agent, "video_path", "") or "",
        AgentFields.SOURCE_TYPE: getattr(agent, "source_type", "rtsp") or "rtsp",
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
        video_path=document.get(AgentFields.VIDEO_PATH, "") or "",
        source_type=document.get(AgentFields.SOURCE_TYPE, "rtsp") or "rtsp",
    )


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
    source_desc = (
        f"video_path={getattr(agent, 'video_path', '')}"
        if (getattr(agent, "source_type", "") or "").strip().lower() == "video_file"
        else f"camera_id={agent.camera_id}"
    )
    logger.info("Saving agent: %s (%s)", agent.name, source_desc)
    coll = get_collection("agents")
    agent_dict = _agent_to_dict_sync(agent)

    # Update existing agent when a valid ObjectId is present.
    if agent.id:
        try:
            oid = ObjectId(agent.id)
            logger.debug("Updating existing agent with ID: %s", agent.id)

            result = coll.update_one(
                {AgentFields.MONGO_ID: oid},
                {"$set": {k: v for k, v in agent_dict.items() if k != AgentFields.MONGO_ID}},
            )

            if result.matched_count > 0:
                doc = coll.find_one({AgentFields.MONGO_ID: oid})
                if doc is None:
                    raise DatabaseError(
                        f"Agent {agent.id} was updated but could not be retrieved",
                        operation="update",
                        retryable=False,
                    )
                logger.info("Agent updated successfully: %s", agent.id)
                return _document_to_agent_sync(doc)

            logger.warning("Agent %s not found for update, inserting instead", agent.id)
        except (InvalidId, ValueError, TypeError) as exc:
            logger.warning("Invalid agent ID %s, inserting instead: %s", agent.id, exc)

    # Insert new agent document.
    if AgentFields.MONGO_ID in agent_dict:
        del agent_dict[AgentFields.MONGO_ID]

    logger.debug("Inserting new agent")
    result = coll.insert_one(agent_dict)
    doc = coll.find_one({AgentFields.MONGO_ID: result.inserted_id})
    if doc is None:
        raise DatabaseError(
            "Agent was created but could not be retrieved",
            operation="insert",
            retryable=False,
        )
    return _document_to_agent_sync(doc)


# ============================================================================
# AGENT SAVING
# ============================================================================

def save_to_db(session_id: str = "default", user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Finalize the agent configuration and save to database.

    Args:
        session_id: Session identifier for state management
        user_id: User ID who owns this agent

    Returns:
        Dict with status, saved flag, agent_id, and agent_name
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

        if agent_state.status == "COLLECTING" and not agent_state.missing_fields:
            print(f"[save_to_db] WARNING: Status is COLLECTING but no missing fields. Recomputing missing fields...")
            from .kb_utils import compute_missing_fields, get_rule
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

        if not user_id:
            print("[save_to_db] ERROR: user_id is not provided")
            return {
                "error": "Cannot save: user_id is required. Agent must be associated with an authenticated user.",
                "status": agent_state.status,
                "saved": False,
                "message": "Authentication error: User ID is required."
            }

        print(f"[save_to_db] Validation passed - proceeding to create agent model")

        run_mode = agent_state.fields.get("run_mode")
        if run_mode and run_mode not in ["continuous", "patrol"]:
            return {
                "error": f"Invalid run_mode: {run_mode}. Only 'continuous' or 'patrol' are allowed.",
                "status": agent_state.status,
                "saved": False,
                "message": f"Invalid run mode: {run_mode}. Please use 'continuous' or 'patrol'."
            }

        from .kb_utils import compute_requires_zone, get_rule
        try:
            rule = get_rule(agent_state.rule_id) if agent_state.rule_id else None
        except ValueError as e:
            return {
                "error": f"Invalid rule_id: {str(e)}",
                "status": agent_state.status,
                "saved": False,
                "message": f"Configuration error: {str(e)}"
            }

        if rule:
            requires_zone = compute_requires_zone(rule, run_mode)
        else:
            requires_zone = False

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

        source_type = (agent_state.fields.get("source_type") or "").strip().lower()
        video_path = (agent_state.fields.get("video_path") or "").strip()
        is_video_file = source_type == "video_file" and video_path
        # Video file agents: no start/end time — agent runs until video ends
        if is_video_file:
            start_time = None
            end_time = None
        payload: Dict[str, Any] = {
            "name": agent_state.fields.get("name", ""),
            "camera_id": agent_state.fields.get("camera_id") or "",
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
            "video_path": video_path if is_video_file else "",
            "source_type": "video_file" if is_video_file else "rtsp",
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

        try:
            print(f"[save_to_db] Attempting to save agent to database...")
            saved_agent = _save_agent_sync(agent)
            print(f"[save_to_db] Repository save() returned: {saved_agent}")
            print(f"[save_to_db] Saved agent ID: {saved_agent.id if saved_agent else 'None'}")
            print(f"[save_to_db] Saved agent name: {saved_agent.name if saved_agent else 'None'}")

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
            logger.exception(f"Unexpected error saving agent: {e}")
            raise DatabaseError(
                f"Failed to save agent to database: {str(e)}",
                operation="save",
                retryable=True,
                user_message="Failed to save agent. Please try again."
            )

        agent_state.status = "SAVED"
        agent_state.saved_agent_id = saved_agent.id if saved_agent else None
        agent_state.saved_agent_name = saved_agent.name if saved_agent else None

        print(f"[save_to_db] Agent state updated to SAVED with agent_id={saved_agent.id}, agent_name={saved_agent.name}")

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
            "flow_diagram": flow_diagram,
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

