# Standard library imports
import json
import uuid
from typing import Optional, List

# External package imports
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
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


def _detect_zone_request(response_text: str, missing_fields: List[str]) -> bool:
    """
    Detect if LLM response is asking for zone input.
    
    Uses both state (missing_fields) and response text for reliability.
    
    Args:
        response_text: The LLM's response text
        missing_fields: List of missing fields from agent state
        
    Returns:
        True if LLM is asking for zone input, False otherwise
    """
    # Primary check: zone must be in missing_fields
    if "zone" not in missing_fields:
        return False
    
    # Secondary check: response contains zone-related language
    zone_keywords = [
        "zone", "area", "region", "draw", "define", "select",
        "polygon", "boundary", "outline", "mark", "highlight",
        "detection zone", "monitoring area", "restricted area"
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
        True if zone is required (requires_zone is True from KB and zone is None)
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
        
        # Zone is required if KB says so AND zone is not yet set
        zone = agent_state.fields.get("zone")
        return bool(requires_zone and zone is None)
    
    # State not initialized - check if agent is asking for zone in response
    # If agent mentions "object_enter_zone" rule, zone is required
    if response_text:
        response_lower = response_text.lower()
        # Check if response mentions object_enter_zone rule
        if "object_enter_zone" in response_lower or "object enter zone" in response_lower:
            # This rule always requires zone
            from ....agents.tools.kb_utils import get_rule, compute_requires_zone
            try:
                rule = get_rule("object_enter_zone")
                requires_zone = compute_requires_zone(rule, "continuous")  # Default to continuous
                return requires_zone
            except (ValueError, KeyError):
                pass
        
        # Also check if zone is in missing_fields (agent might have initialized but we're checking before state update)
        if "zone" in agent_state.missing_fields:
            return True
    
    return False


class ChatWithAgentUseCase:
    """Use case for chatting with the agent chatbot"""
    
    # Shared session service across all instances (singleton pattern)
    _shared_session_service: Optional[InMemorySessionService] = None
    
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
        
        # Store session mappings: {session_id: (adk_session, agent_instance)}
        # Note: This is per-instance, but sessions are stored in the shared session_service
        self._sessions: dict[str, tuple] = {}
        # Store user_id for each session
        self._session_user_map: dict[str, str] = {}
        
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
        
        # Get or create session
        # If session_id is provided, try to get existing session first
        # Otherwise, create a new session
        session_id = request.session_id
        adk_session = None
        
        # If session_id is provided, try to get existing session from ADK session service
        if session_id:
            try:
                adk_session = await self.session_service.get_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id
                )
            except Exception as e:
                # If get_session fails, log and continue to create new session
                # In production, use proper logging instead of print
                adk_session = None
        
        # If session doesn't exist, create a new one
        if not adk_session:
            if not session_id:
                session_id = self._create_session_id()
            
            try:
                adk_session = await self.session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id
                )
            except Exception as e:
                # If create fails (e.g., session already exists), try to get it
                # In production, use proper logging instead of print
                adk_session = await self.session_service.get_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id
                )
                if not adk_session:
                    # If still not found, create with a new session_id
                    session_id = self._create_session_id()
                    adk_session = await self.session_service.create_session(
                        app_name=app_name,
                        user_id=user_id,
                        session_id=session_id
                    )
        
        # Get or create agent for this session
        # Always create a new agent to ensure we have the latest tool wrappings
        # (This ensures FunctionTool wrappings are always used)
        # Note: Instruction is now dynamic and rebuilds automatically on each LLM call
        agent = create_agent_for_session(session_id=session_id, user_id=user_id)
        
        # Store mapping between ADK session.id and our internal session_id in ADK session state
        # This allows the dynamic instruction provider to find the right state
        if "internal_session_id" not in adk_session.state:
            adk_session.state["internal_session_id"] = session_id
        
        self._sessions[session_id] = (adk_session, agent)
        self._session_user_map[session_id] = user_id
        
        # Create runner for this agent with required session service
        runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=self.session_service
        )
        
        # Handle zone_data if provided from UI
        user_message = request.message
        if request.zone_data:
            # Merge zone data into message as JSON
            # LLM will extract and set it via set_field_value
            # Use string concatenation instead of f-string to avoid format specifier issues with JSON
            zone_json = json.dumps(request.zone_data)
            user_message = user_message + "\n\nZone data: " + zone_json
        
        # Handle camera_id if provided from UI
        # Strategy: 
        # - If state is NOT initialized: Include camera_id in message so LLM can extract it during initialization
        # - If state IS initialized: Set camera_id directly in state
        if request.camera_id:
            from ....agents.session_state.agent_state import get_agent_state
            agent_state = get_agent_state(session_id)
            
            if agent_state.rule_id:
                # State is already initialized - set camera_id directly
                agent_state.fields["camera_id"] = request.camera_id
                # Remove camera_id from missing_fields if it's there
                if "camera_id" in agent_state.missing_fields:
                    agent_state.missing_fields.remove("camera_id")
            else:
                # State not initialized yet - include camera_id in message
                # LLM will extract it when user provides agent creation intent
                # This way greetings work normally without premature state setting
                user_message = user_message + f"\n\nCamera ID: {request.camera_id}"
        
        # Create Content object from user message
        user_content = types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )
        
        # Run agent with user message
        try:
            # run_async returns an async generator of events
            # ADK emits multiple events: model text, tool calls, model text again
            # We only want the FINAL model response, not intermediate ones
            final_response_text = ""
            last_model_response = ""
            
            async for event in runner.run_async(
                user_id=user_id,
                session_id=adk_session.id,
                new_message=user_content
            ):
                # Check if this is a final model response event
                # Only collect text from final responses to avoid mixing messages
                if hasattr(event, 'is_final_response') and event.is_final_response():
                    # Extract text from final response events only
                    if hasattr(event, 'content') and event.content:
                        if isinstance(event.content, types.Content):
                            # Extract text from parts
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    final_response_text += part.text
                        elif hasattr(event.content, 'parts'):
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    final_response_text += part.text
                    # Also check for direct text attributes
                    elif hasattr(event, 'text') and event.text:
                        final_response_text += event.text
                # Track the last model response (even if not marked final)
                # Some models might not mark final correctly
                # Author is the agent name (e.g., "main_agent"), not "model"
                # Also check for any non-user author (could be agent name)
                elif hasattr(event, 'author') and event.author != "user" and event.author:
                    # Extract text for tracking
                    event_text = ""
                    if hasattr(event, 'content') and event.content:
                        if isinstance(event.content, types.Content):
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    event_text += part.text
                        elif hasattr(event.content, 'parts'):
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    event_text += part.text
                    elif hasattr(event, 'text') and event.text:
                        event_text = event.text
                    
                    if event_text:
                        last_model_response = event_text
            
            # Use final response if available, otherwise use last model response
            response_to_return = final_response_text.strip() if final_response_text.strip() else last_model_response.strip()
            
            # Get current agent state for zone signals
            from ....agents.session_state.agent_state import get_agent_state
            agent_state = get_agent_state(session_id)
            
            # Compute zone signals
            # Pass response text to help infer zone requirement if state not initialized
            response_text_for_zone = response_to_return if response_to_return else last_model_response
            zone_required = _compute_zone_required(agent_state, response_text_for_zone)
            awaiting_zone_input = _detect_zone_request(
                response_text_for_zone,
                agent_state.missing_fields
            )
            
            return ChatMessageResponse(
                response=response_to_return if response_to_return else "I apologize, but I didn't receive a proper response.",
                session_id=session_id,
                status="success",
                zone_required=zone_required,
                awaiting_zone_input=awaiting_zone_input,
            )
        except Exception as e:
            # On error, still compute zone signals for UI consistency
            try:
                from ....agents.session_state.agent_state import get_agent_state
                agent_state = get_agent_state(session_id)
                # Pass empty string for response text in error case
                zone_required = _compute_zone_required(agent_state, "")
                awaiting_zone_input = False  # Error state, not asking for input
            except:
                zone_required = False
                awaiting_zone_input = False
            
            return ChatMessageResponse(
                response=f"I encountered an error: {str(e)}. Please try again.",
                session_id=session_id,
                status="error",
                zone_required=zone_required,
                awaiting_zone_input=awaiting_zone_input,
            )
    
    def _create_session_id(self) -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())
