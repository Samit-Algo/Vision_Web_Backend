# Standard library imports
import json
import logging
import re
import threading
import uuid
from typing import Optional, List, Tuple, AsyncGenerator, Dict, Any

# External package imports
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types

# Local application imports
from ...dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ....agents.main_agent import create_agent_for_session
from ....domain.repositories.agent_repository import AgentRepository
from ....domain.repositories.camera_repository import CameraRepository
from ....domain.repositories.device_repository import DeviceRepository
from ....agents.tools.save_to_db_tool import (
    set_agent_repository,
    set_camera_repository,
    set_device_repository,
    set_jetson_client
)
from ....infrastructure.external.agent_client import AgentClient


logger = logging.getLogger(__name__)


def _detect_zone_request(response_text: str, missing_fields: List[str], agent_state=None) -> bool:
    """
    Detect if LLM response is asking for zone input.
    
    Uses both state (missing_fields) and response text for reliability.
    
    Args:
        response_text: The LLM's response text
        missing_fields: List of missing fields from agent state
        agent_state: Optional agent state to check rule context
        
    Returns:
        True if LLM is asking for zone input, False otherwise
    """
    # Primary check: zone must be in missing_fields
    if "zone" not in missing_fields:
        # Special case: For counting rules with crossing intent, offer zone drawing
        # even though zone is optional
        if agent_state and agent_state.rule_id in ["class_count", "box_count"]:
            # Check if this is a confirmation stage (all required fields collected)
            from ....agents.tools.kb_utils import get_rule, compute_requires_zone
            try:
                rule = get_rule(agent_state.rule_id)
                run_mode = agent_state.fields.get("run_mode", "continuous")
                requires_zone = compute_requires_zone(rule, run_mode)
                zone = agent_state.fields.get("zone")
                # Check if zone is empty
                zone_is_empty = not zone or (isinstance(zone, (dict, list)) and not zone)
                
                # If zone is optional (not required), check if zone is empty
                if not requires_zone and zone_is_empty and not missing_fields:
                    # Check if user intent or response mentions crossing/line drawing
                    response_lower = response_text.lower()
                    crossing_keywords = [
                        "cross", "crossing", "line", "draw", "gate", "door", 
                        "entrance", "exit", "entry", "boundary"
                    ]
                    if any(keyword in response_lower for keyword in crossing_keywords):
                        # This is crossing detection - should offer zone drawing
                        return True
            except (ValueError, KeyError):
                pass
        
        return False
    
    # Secondary check: response contains zone-related language
    zone_keywords = [
        "zone", "area", "region", "draw", "define", "select",
        "polygon", "boundary", "outline", "mark", "highlight",
        "detection zone", "monitoring area", "restricted area",
        "line", "cross", "crossing"  # Add crossing-related keywords
    ]
    
    response_lower = response_text.lower()
    return any(keyword in response_lower for keyword in zone_keywords)


def _compute_zone_required(agent_state, response_text: str = "") -> bool:
    """
    Compute if zone is required based on state and knowledge base.
    Deterministic - never manually set.
    
    Args:
        agent_state: The current agent state
        response_text: Optional response text to infer rule if state not initialized
        
    Returns:
        True if zone is required (requires_zone is True from KB and zone is None/empty)
    """
    # If state is initialized, compute from KB
    if agent_state.rule_id:
        # Compute requires_zone from knowledge base (derived state, not stored)
        from ....agents.tools.kb_utils import get_rule, compute_requires_zone
        try:
            rule = get_rule(agent_state.rule_id)
            run_mode = agent_state.fields.get("run_mode", "continuous")
            requires_zone = compute_requires_zone(rule, run_mode)
        except (ValueError, KeyError):
            # Rule not found or error - assume zone not required
            requires_zone = False
        
        # Check if zone is empty or not set
        zone = agent_state.fields.get("zone")
        zone_is_empty = not zone or (isinstance(zone, (dict, list)) and not zone)
        
        # Zone is required if KB says so AND zone is not yet set/empty
        zone_required_by_kb = bool(requires_zone and zone_is_empty)
        
        # Special case: For counting rules, treat zone as "needed for UI" if zone is empty
        if not zone_required_by_kb and agent_state.rule_id in ["class_count", "box_count"]:
            # Zone is optional but we should show UI if zone is empty
            if zone_is_empty:
                zone_required_by_kb = True
        
        return zone_required_by_kb
    
    # State not initialized (no rule_id) - do NOT infer zone from response text.
    # Listing rules (e.g. "get all rules") produces text that mentions many rules
    # (object enter zone, class count, etc.), which would incorrectly set zone_required=True.
    # Only require zone when the agent has explicitly asked for it (zone in missing_fields).
    if "zone" in agent_state.missing_fields:
        return True

    return False


class ChatWithAgentUseCase:
    """Use case for chatting with the agent chatbot"""
    
    # Shared session service across all instances (singleton pattern)
    _shared_session_service: Optional[InMemorySessionService] = None
    # Shared agent cache across all instances (singleton pattern)
    # Keyed by (user_id, session_id) to avoid cross-user leakage.
    _shared_agent_cache: dict[Tuple[str, str], LlmAgent] = {}
    _shared_agent_cache_lock = threading.Lock()
    
    def __init__(
        self,
        agent_repository: Optional[AgentRepository] = None,
        camera_repository: Optional[CameraRepository] = None,
        device_repository: Optional[DeviceRepository] = None,
        jetson_client: Optional[AgentClient] = None,
    ) -> None:
        # Use shared session service to persist sessions across requests
        if ChatWithAgentUseCase._shared_session_service is None:
            ChatWithAgentUseCase._shared_session_service = InMemorySessionService()
        self.session_service = ChatWithAgentUseCase._shared_session_service
        
        # Set up repositories and client for tools if provided
        if agent_repository:
            set_agent_repository(agent_repository)
        if camera_repository:
            set_camera_repository(camera_repository)
        if device_repository:
            set_device_repository(device_repository)
        if jetson_client:
            set_jetson_client(jetson_client)
    
    async def execute(
        self, 
        request: ChatMessageRequest,
        user_id: Optional[str] = None
    ) -> ChatMessageResponse:
        """
        Send a message to the agent and get a response.
        
        Args:
            request: Chat message request
            user_id: Optional user ID for saving agents
            
        Returns:
            ChatMessageResponse with agent's response
        """
        # Ensure we have a user_id (required by ADK session service)
        if not user_id:
            user_id = "anonymous"  # Default user ID if not provided
        
        # App name for ADK session service
        app_name = "vision-agent-chat"
        
        adk_session, session_id = await self._get_or_create_adk_session(
            app_name=app_name,
            user_id=user_id,
            session_id=request.session_id,
        )

        agent = self._get_or_create_agent_cached(user_id=user_id, session_id=session_id)

        # Keep mapping between ADK session and internal session_id for downstream lookups
        if "internal_session_id" not in adk_session.state:
            adk_session.state["internal_session_id"] = session_id

        runner = Runner(app_name=app_name, agent=agent, session_service=self.session_service)

        user_message = self._build_user_message(
            request=request, session_id=session_id, user_id=user_id
        )
        user_content = types.Content(role="user", parts=[types.Part(text=user_message)])

        try:
            final_response_text, last_model_response = await self._run_agent_and_collect_text(
                runner=runner,
                user_id=user_id,
                adk_session_id=adk_session.id,
                user_content=user_content,
            )

            response_to_return = (
                final_response_text.strip() if final_response_text.strip() else last_model_response.strip()
            )

            return await self._finalize_chat_response(
                response_to_return=response_to_return,
                last_model_response=last_model_response,
                session_id=session_id,
                adk_session=adk_session,
                status="success",
            )
        except Exception as e:
            logger.exception("[chat] execute failed")
            return await self._finalize_chat_response(
                response_to_return="",
                last_model_response="",
                session_id=session_id,
                adk_session=adk_session,
                status="error",
                error=e,
            )

    async def stream_execute(
        self,
        *,
        request: ChatMessageRequest,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream agent output as structured events.

        Yields dict events:
        - {"event": "meta", "data": {...}}
        - {"event": "token", "data": {...}}
        - {"event": "done", "data": ChatMessageResponse-compatible dict}
        - {"event": "error", "data": {...}}
        """
        if not user_id:
            user_id = "anonymous"

        app_name = "vision-agent-chat"

        adk_session, session_id = await self._get_or_create_adk_session(
            app_name=app_name,
            user_id=user_id,
            session_id=request.session_id,
        )

        agent = self._get_or_create_agent_cached(user_id=user_id, session_id=session_id)

        if "internal_session_id" not in getattr(adk_session, "state", {}):
            try:
                adk_session.state["internal_session_id"] = session_id
            except Exception:
                pass

        runner = Runner(app_name=app_name, agent=agent, session_service=self.session_service)

        user_message = self._build_user_message(
            request=request, session_id=session_id, user_id=user_id
        )
        user_content = types.Content(role="user", parts=[types.Part(text=user_message)])

        # Let the client know the session_id immediately so UI can persist it mid-stream
        yield {"event": "meta", "data": {"session_id": session_id}}

        try:
            final_response_text = ""
            last_model_response = ""
            async for chunk in self._run_agent_stream(
                runner=runner,
                user_id=user_id,
                adk_session_id=adk_session.id,
                user_content=user_content,
            ):
                if chunk.get("type") == "token":
                    delta = chunk.get("delta") or ""
                    if delta:
                        yield {"event": "token", "data": {"delta": delta}}
                elif chunk.get("type") == "result":
                    final_response_text = chunk.get("final_response_text") or ""
                    last_model_response = chunk.get("last_model_response") or ""

            response_to_return = (
                final_response_text.strip() if final_response_text.strip() else last_model_response.strip()
            )

            final_response = await self._finalize_chat_response(
                response_to_return=response_to_return,
                last_model_response=last_model_response,
                session_id=session_id,
                adk_session=adk_session,
                status="success",
            )
            yield {"event": "done", "data": final_response.model_dump()}
        except Exception as e:
            logger.exception("[chat] stream_execute failed")
            final_response = await self._finalize_chat_response(
                response_to_return="",
                last_model_response="",
                session_id=session_id,
                adk_session=adk_session,
                status="error",
                error=e,
            )
            yield {"event": "error", "data": {"message": str(e)}}
            yield {"event": "done", "data": final_response.model_dump()}

    async def _finalize_chat_response(
        self,
        *,
        response_to_return: str,
        last_model_response: str,
        session_id: str,
        adk_session: object,
        status: str,
        error: Optional[Exception] = None,
    ) -> ChatMessageResponse:
        """Apply flow diagram + zone flags and build ChatMessageResponse."""
        if status != "success":
            # On error, still compute zone signals for UI consistency
            try:
                from ....agents.session_state.agent_state import get_agent_state

                agent_state = get_agent_state(session_id)
                zone_required = _compute_zone_required(agent_state, "")
                awaiting_zone_input = False
                camera_id = agent_state.fields.get("camera_id")
            except Exception:
                zone_required = False
                awaiting_zone_input = False
                camera_id = None

            msg = f"I encountered an error: {str(error)}. Please try again." if error else "I encountered an error."
            return ChatMessageResponse(
                response=msg,
                session_id=session_id,
                status="error",
                camera_id=camera_id,
                zone_required=zone_required,
                awaiting_zone_input=awaiting_zone_input,
                frame_snapshot_url=None,
                zone_type=None,
            )

        # Prefer response_to_return; fall back to last_model_response
        response_text = response_to_return.strip() if response_to_return else last_model_response.strip()
        if not response_text:
            response_text = "I apologize, but I didn't receive a proper response."

        # Get current agent state for zone signals / flow diagram.
        # IMPORTANT: tools store state keyed by our internal session_id, but ADK can also key by adk_session.id.
        # We defensively check both so flow diagram always attaches after a real save.
        from ....agents.session_state.agent_state import get_agent_state

        agent_state_primary = get_agent_state(session_id)
        agent_state_adk = get_agent_state(getattr(adk_session, "id", session_id))

        agent_state = agent_state_primary
        if (not agent_state.saved_agent_id) and agent_state_adk.saved_agent_id:
            agent_state = agent_state_adk

        # Check if agent was just saved and attach flow diagram data
        flow_diagram_data = None
        if agent_state.saved_agent_id:
            try:
                from ....di.container import get_container

                container = get_container()
                agent_repository = container.get(AgentRepository)
                saved_agent = await agent_repository.find_by_id(agent_state.saved_agent_id)

                if saved_agent:
                    from ....agents.tools.flow_diagram_utils import generate_agent_flow_diagram

                    # Generate raw diagram data (single generation, frontend transforms)
                    flow_diagram_data = generate_agent_flow_diagram(saved_agent)

                    logger.debug(
                        "[chat] flow diagram generated (nodes=%d, links=%d)",
                        len(flow_diagram_data.get("nodes", [])),
                        len(flow_diagram_data.get("links", [])),
                    )

                    # Add a simple text indicator for user context
                    response_text += "\n\n---\n\n## Processing Flow Diagram\n\n*Your agent's processing flow is shown below.*"
            except Exception:
                logger.exception("[chat] failed to generate flow diagram")

        response_text_for_zone = response_text if response_text else last_model_response
        zone_required = _compute_zone_required(agent_state, response_text_for_zone)
        awaiting_zone_input = _detect_zone_request(
            response_text_for_zone,
            agent_state.missing_fields,
            agent_state,  # Pass agent_state for crossing detection logic
        )

        # Do not show zone UI for rules that do not support zones (e.g. sleep_detection).
        if agent_state.rule_id:
            from ....agents.tools.kb_utils import get_rule
            try:
                rule = get_rule(agent_state.rule_id)
                zone_support = rule.get("zone_support", {})
                if zone_support.get("supported", True) is False:
                    zone_required = False
                    awaiting_zone_input = False
            except (ValueError, KeyError):
                pass

        # Generate snapshot URL and zone type if zone drawing is needed.
        # Use camera_id from either session state (tools may key by session_id or adk_session.id).
        camera_id = (
            agent_state_primary.fields.get("camera_id")
            or agent_state_adk.fields.get("camera_id")
        )
        frame_snapshot_url = None
        zone_type = None
        
        if awaiting_zone_input or zone_required:
            print(f"[CHAT_RESPONSE] 🎯 Zone UI needed: awaiting_zone_input={awaiting_zone_input}, zone_required={zone_required}")
            print(f"[CHAT_RESPONSE] 📷 Camera ID from agent_state.fields: {camera_id}")
            
            if camera_id:
                # Generate snapshot URL for the camera
                frame_snapshot_url = f"/api/v1/cameras/{camera_id}/snapshot"
                print(f"[CHAT_RESPONSE] ✅ Frame snapshot URL set: {frame_snapshot_url}")
                
                # Determine zone type from rule
                if agent_state.rule_id:
                    from ....agents.tools.kb_utils import get_rule
                    try:
                        rule = get_rule(agent_state.rule_id)
                        zone_support = rule.get("zone_support", {})
                        zone_type = zone_support.get("zone_type", "polygon")  # Default to polygon
                        print(f"[CHAT_RESPONSE] 📐 Zone type from rule: {zone_type}")
                    except (ValueError, KeyError):
                        zone_type = "polygon"  # Default
                        print(f"[CHAT_RESPONSE] ⚠️  Could not get zone_type from rule, defaulting to polygon")
            else:
                print(f"[CHAT_RESPONSE] ⚠️  Camera ID is None - cannot set frame_snapshot_url for zone editor")
                print(f"[CHAT_RESPONSE] 📋 Agent state fields: {list(agent_state.fields.keys())}")
                print(f"[CHAT_RESPONSE] 📋 Missing fields: {agent_state.missing_fields}")

        return ChatMessageResponse(
            response=response_text,
            session_id=session_id,
            status="success",
            camera_id=camera_id,  # Always include camera_id if available
            zone_required=zone_required,
            awaiting_zone_input=awaiting_zone_input,
            frame_snapshot_url=frame_snapshot_url,
            zone_type=zone_type,
            flow_diagram_data=flow_diagram_data,
        )

    async def _run_agent_stream(
        self,
        *,
        runner: Runner,
        user_id: str,
        adk_session_id: str,
        user_content: types.Content,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run ADK and yield streaming deltas, then a final result chunk.

        Yields:
        - {"type": "token", "delta": "..."} for incremental text updates
        - {"type": "result", "final_response_text": "...", "last_model_response": "..."} at the end
        """
        final_response_text = ""
        last_model_response = ""
        emitted_so_far = ""

        async for event in runner.run_async(
            user_id=user_id,
            session_id=adk_session_id,
            new_message=user_content,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            if hasattr(event, "is_final_response") and event.is_final_response():
                if hasattr(event, "content") and event.content:
                    if isinstance(event.content, types.Content):
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                final_response_text += part.text
                    elif hasattr(event.content, "parts"):
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                final_response_text += part.text
                elif hasattr(event, "text") and event.text:
                    final_response_text += event.text
                continue

            if hasattr(event, "author") and event.author != "user" and event.author:
                event_text = ""
                if hasattr(event, "content") and event.content:
                    if isinstance(event.content, types.Content):
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                event_text += part.text
                    elif hasattr(event.content, "parts"):
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                event_text += part.text
                elif hasattr(event, "text") and event.text:
                    event_text = event.text

                if not event_text:
                    continue

                # Heuristic: if event_text is a growing prefix, only emit the delta.
                delta = event_text
                if emitted_so_far and event_text.startswith(emitted_so_far):
                    delta = event_text[len(emitted_so_far) :]
                elif not emitted_so_far and event_text:
                    # first emission: treat as delta as-is
                    delta = event_text

                # Track latest full text
                last_model_response = event_text
                emitted_so_far = event_text

                if delta:
                    yield {"type": "token", "delta": delta}

        yield {
            "type": "result",
            "final_response_text": final_response_text,
            "last_model_response": last_model_response,
            "emitted_text": emitted_so_far,
        }

    async def _get_or_create_adk_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str],
    ) -> Tuple[object, str]:
        """Get an existing ADK session or create a new one."""
        adk_session = None

        if session_id:
            try:
                adk_session = await self.session_service.get_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                )
            except Exception:
                logger.debug("[chat] get_session failed; will create a new session", exc_info=True)
                adk_session = None

        if adk_session:
            return adk_session, session_id  # type: ignore[return-value]

        if not session_id:
            session_id = self._create_session_id()

        try:
            adk_session = await self.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
        except Exception:
            # If create fails (e.g. already exists), try to get it again; otherwise create with a new ID.
            logger.debug("[chat] create_session failed; retrying get_session", exc_info=True)
            adk_session = await self.session_service.get_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if not adk_session:
                session_id = self._create_session_id()
                adk_session = await self.session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                )

        return adk_session, session_id  # type: ignore[return-value]

    def _get_or_create_agent_cached(self, *, user_id: str, session_id: str) -> LlmAgent:
        """Get or create the ADK agent instance for this (user_id, session_id)."""
        cache_key = (user_id, session_id)
        with ChatWithAgentUseCase._shared_agent_cache_lock:
            agent = ChatWithAgentUseCase._shared_agent_cache.get(cache_key)
            if agent is None:
                agent = create_agent_for_session(session_id=session_id, user_id=user_id)
                ChatWithAgentUseCase._shared_agent_cache[cache_key] = agent
            return agent

    def _build_user_message(
        self,
        *,
        request: ChatMessageRequest,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> str:
        """Build the final user message, merging optional UI-provided fields."""
        user_message = request.message

        if request.zone_data:
            # If agent state is already initialized, set zone deterministically (do not rely on LLM parsing).
            # Otherwise, fall back to appending zone JSON into the user message for rule selection stage.
            try:
                from ....agents.session_state.agent_state import get_agent_state

                agent_state = get_agent_state(session_id)
                if agent_state.rule_id:
                    zone_payload = request.zone_data
                    if isinstance(zone_payload, dict) and "zone" in zone_payload:
                        zone_payload = zone_payload.get("zone")
                    agent_state.fields["zone"] = zone_payload
                    if "zone" in agent_state.missing_fields:
                        agent_state.missing_fields.remove("zone")
                else:
                    zone_json = json.dumps(request.zone_data)
                    user_message = user_message + "\n\nZone data: " + zone_json
            except Exception:
                zone_json = json.dumps(request.zone_data)
                user_message = user_message + "\n\nZone data: " + zone_json

        if request.camera_id:
            from ....agents.session_state.agent_state import get_agent_state

            agent_state = get_agent_state(session_id)
            if agent_state.rule_id:
                agent_state.fields["camera_id"] = request.camera_id
                if "camera_id" in agent_state.missing_fields:
                    agent_state.missing_fields.remove("camera_id")
            else:
                user_message = user_message + f"\n\nCamera ID: {request.camera_id}"

        # If user typed a camera ID or camera name in the message but the LLM didn't
        # call set_field_value_wrapper, persist it here so frame_snapshot_url can be set.
        try:
            from ....agents.session_state.agent_state import get_agent_state
            from ....agents.tools.camera_selection_tool import resolve_camera as resolve_camera_impl

            agent_state = get_agent_state(session_id)
            if agent_state.rule_id and not agent_state.fields.get("camera_id"):
                raw = (user_message or "").strip()
                # 1) Match explicit camera ID: CAM- + alphanumeric
                cam_match = re.match(r"^CAM-[A-Za-z0-9]+$", raw)
                if not cam_match:
                    cam_match = re.search(r"CAM-[A-Za-z0-9]+", raw)
                if cam_match:
                    agent_state.fields["camera_id"] = cam_match.group(0)
                    if "camera_id" in agent_state.missing_fields:
                        agent_state.missing_fields.remove("camera_id")
                else:
                    # 2) Try resolving as camera name (e.g. "Test", "Camera Id : Test", "Front Gate")
                    candidate = None
                    for pattern in (
                        r"(?i)camera\s*id\s*[:\s]+\s*(\S+(?:\s+\S+)*)",
                        r"(?i)camera\s*name\s*[:\s]+\s*(\S+(?:\s+\S+)*)",
                        r"(?i)camera\s*[:\s]+\s*(\S+(?:\s+\S+)*)",
                    ):
                        m = re.search(pattern, raw)
                        if m:
                            candidate = m.group(1).strip()
                            break
                    if not candidate and raw and len(raw) <= 80 and not re.search(r"^\d+$", raw):
                        candidate = raw.strip()
                    if candidate and user_id:
                        result = resolve_camera_impl(
                            name_or_id=candidate,
                            user_id=user_id,
                            session_id=session_id,
                        )
                        if result.get("status") == "exact_match" and result.get("camera_id"):
                            agent_state.fields["camera_id"] = result["camera_id"]
                            if "camera_id" in agent_state.missing_fields:
                                agent_state.missing_fields.remove("camera_id")
                            logger.info(
                                "[chat] Resolved camera name %r to camera_id=%s",
                                candidate,
                                result["camera_id"],
                            )
        except Exception:
            pass

        return user_message

    async def _run_agent_and_collect_text(
        self,
        *,
        runner: Runner,
        user_id: str,
        adk_session_id: str,
        user_content: types.Content,
    ) -> Tuple[str, str]:
        """Run ADK and return (final_response_text, last_model_response)."""
        final_response_text = ""
        last_model_response = ""

        async for event in runner.run_async(
            user_id=user_id,
            session_id=adk_session_id,
            new_message=user_content,
        ):
            if hasattr(event, "is_final_response") and event.is_final_response():
                if hasattr(event, "content") and event.content:
                    if isinstance(event.content, types.Content):
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                final_response_text += part.text
                    elif hasattr(event.content, "parts"):
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                final_response_text += part.text
                elif hasattr(event, "text") and event.text:
                    final_response_text += event.text
            elif hasattr(event, "author") and event.author != "user" and event.author:
                event_text = ""
                if hasattr(event, "content") and event.content:
                    if isinstance(event.content, types.Content):
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                event_text += part.text
                    elif hasattr(event.content, "parts"):
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                event_text += part.text
                elif hasattr(event, "text") and event.text:
                    event_text = event.text

                if event_text:
                    last_model_response = event_text

        return final_response_text, last_model_response
    
    def _create_session_id(self) -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())
