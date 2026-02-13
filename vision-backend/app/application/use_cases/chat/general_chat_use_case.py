"""Use case for general chat with a simple Google ADK agent."""
import json
import re
import uuid
from typing import Optional, AsyncGenerator, Dict, Any
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types

from ...dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ....agents.general_chat import create_general_chat_agent


_EVENT_IMAGE_URL_RE = re.compile(r"https?://[^\s)]+/api/v1/events/[^\s)]+/image")


def _ensure_event_images_markdown(text: str) -> str:
    """Append deterministic markdown image tags for raw event image URLs."""
    if not text:
        return text

    urls = []
    for url in _EVENT_IMAGE_URL_RE.findall(text):
        if url not in urls:
            urls.append(url)

    if not urls:
        return text

    missing_urls = []
    for url in urls:
        if f"![Evidence]({url})" in text:
            continue
        if f"]({url})" in text:
            continue
        missing_urls.append(url)

    if not missing_urls:
        return text

    return text + "\n\n" + "\n".join(f"![Evidence]({url})" for url in missing_urls)


class GeneralChatUseCase:
    """Use case for general chat conversations."""
    
    # Shared session service across all instances
    _shared_session_service: Optional[InMemorySessionService] = None
    
    def __init__(self) -> None:
        # Use shared session service to persist sessions across requests
        if GeneralChatUseCase._shared_session_service is None:
            GeneralChatUseCase._shared_session_service = InMemorySessionService()
        self.session_service = GeneralChatUseCase._shared_session_service
        
        # Store session mappings: {session_id: (adk_session, agent_instance)}
        self._sessions: dict[str, tuple] = {}
        self._session_user_map: dict[str, str] = {}
    
    async def execute(
        self,
        request: ChatMessageRequest,
        user_id: Optional[str] = None
    ) -> ChatMessageResponse:
        """
        Send a message to the general chat agent and get a response.
        
        Args:
            request: Chat message request with message and optional session_id
            user_id: Optional user ID for session management
            
        Returns:
            ChatMessageResponse with agent's response and session_id
        """
        # Ensure we have a user_id (required by ADK session service)
        if not user_id:
            user_id = "anonymous"
        
        # App name for ADK session service
        app_name = "general-chat"
        
        # Get or create session
        session_id = request.session_id
        adk_session = None
        
        # Try to get existing session if session_id is provided
        if session_id:
            try:
                adk_session = await self.session_service.get_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id
                )
            except Exception:
                adk_session = None
        
        # Create new session if doesn't exist
        if not adk_session:
            if not session_id:
                session_id = self._create_session_id()
            
            try:
                adk_session = await self.session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id
                )
            except Exception:
                # If create fails, try to get it
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
        
        # Import the tools for wrapping
        from ....agents.general_chat.tools import (
            list_my_cameras, find_camera, check_camera_health, 
            get_recent_detections, get_deployed_agents_summary, get_event_details
        )
        
        # Create tool wrappers with current user context
        def list_cameras_wrapper():
            """List all cameras owned by the current user."""
            return list_my_cameras(user_id=user_id)
            
        def find_camera_wrapper(name_or_id: str):
            """Find a specific camera by name or ID for the current user."""
            return find_camera(name_or_id=name_or_id, user_id=user_id)

        def check_camera_health_wrapper(camera_id: str):
            """Check if a specific camera is online and working. Requires a camera_id."""
            return check_camera_health(camera_id=camera_id, user_id=user_id)

        def get_detections_wrapper(camera_id: str = "", days_ago: int = 0, limit: int = 10):
            """
            Search for vision detections/events. 
            - camera_id: (Optional) ID of a specific camera.
            - days_ago: 0 for today, 1 for yesterday, 2 for two days ago, etc.
            - limit: How many events to show.
            """
            return get_recent_detections(user_id=user_id, camera_id=camera_id, days_ago=days_ago, limit=limit)
            
        def get_deployed_agents_wrapper():
            """Get a summary of all vision agents currently deployed by the current user."""
            return get_deployed_agents_summary(user_id=user_id)
            
        def get_event_details_wrapper(event_id: str):
            """Get technical details and metadata for a specific event by ID."""
            return get_event_details(user_id=user_id, event_id=event_id)
            
        # Create agent instance for general chat with wrapped tools
        agent_tools = [
            list_cameras_wrapper, find_camera_wrapper, check_camera_health_wrapper,
            get_detections_wrapper, get_deployed_agents_wrapper, get_event_details_wrapper
        ]
        
        # Add dynamic time context
        from ....agents.utils.time_context import get_current_time_context
        def instruction_provider(context):
            return get_current_time_context()
            
        agent = create_general_chat_agent(tools=agent_tools, instruction=instruction_provider)
        
        # Store session mapping
        self._sessions[session_id] = (adk_session, agent)
        self._session_user_map[session_id] = user_id
        
        # Create runner for this agent
        runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=self.session_service
        )
        
        # Create Content object from user message
        user_content = types.Content(
            role="user",
            parts=[types.Part(text=request.message)]
        )
        
        # Run agent with user message
        try:
            final_response_text = ""
            last_model_response = ""
            
            async for event in runner.run_async(
                user_id=user_id,
                session_id=adk_session.id,
                new_message=user_content
            ):
                # Check if this is a final model response event
                if hasattr(event, 'is_final_response') and event.is_final_response():
                    if hasattr(event, 'content') and event.content:
                        if isinstance(event.content, types.Content):
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    final_response_text += part.text
                        elif hasattr(event.content, 'parts'):
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    final_response_text += part.text
                    elif hasattr(event, 'text') and event.text:
                        final_response_text += event.text
                # Track the last model response
                elif hasattr(event, 'author') and event.author != "user" and event.author:
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
            
            response_to_return = response_to_return if response_to_return else "I apologize, but I didn't receive a proper response."
            response_to_return = _ensure_event_images_markdown(response_to_return)

            return ChatMessageResponse(
                response=response_to_return,
                session_id=session_id,
                status="success"
            )
        except Exception as e:
            return ChatMessageResponse(
                response=f"I encountered an error: {str(e)}. Please try again.",
                session_id=session_id,
                status="error"
            )

    async def stream_execute(
        self,
        *,
        request: ChatMessageRequest,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream general chat output as structured events.

        Yields dict events:
        - {"event": "meta", "data": {"session_id": "..."}}
        - {"event": "token", "data": {"delta": "..."}}
        - {"event": "done", "data": ChatMessageResponse-compatible dict}
        - {"event": "error", "data": {"message": "..."}}
        """
        if not user_id:
            user_id = "anonymous"

        app_name = "general-chat"

        # Get or create session (same logic as execute)
        session_id = request.session_id
        adk_session = None
        if session_id:
            try:
                adk_session = await self.session_service.get_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                )
            except Exception:
                adk_session = None

        if not adk_session:
            if not session_id:
                session_id = self._create_session_id()
            try:
                adk_session = await self.session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                )
            except Exception:
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

        # Import the tools for wrapping
        from ....agents.general_chat.tools import (
            list_my_cameras, find_camera, check_camera_health, 
            get_recent_detections, get_deployed_agents_summary, get_event_details
        )
        
        # Create tool wrappers with current user context
        def list_cameras_wrapper():
            """List all cameras owned by the current user."""
            return list_my_cameras(user_id=user_id)
            
        def find_camera_wrapper(name_or_id: str):
            """Find a specific camera by name or ID for the current user."""
            return find_camera(name_or_id=name_or_id, user_id=user_id)

        def check_camera_health_wrapper(camera_id: str):
            """Check if a specific camera is online and working. Requires a camera_id."""
            return check_camera_health(camera_id=camera_id, user_id=user_id)

        def get_detections_wrapper(camera_id: str = "", days_ago: int = 0, limit: int = 10):
            """
            Search for vision detections/events. 
            - camera_id: (Optional) ID of a specific camera.
            - days_ago: 0 for today, 1 for yesterday, 2 for two days ago, etc.
            - limit: How many events to show.
            """
            return get_recent_detections(user_id=user_id, camera_id=camera_id, days_ago=days_ago, limit=limit)
            
        def get_deployed_agents_wrapper():
            """Get a summary of all vision agents currently deployed by the current user."""
            return get_deployed_agents_summary(user_id=user_id)
            
        def get_event_details_wrapper(event_id: str):
            """Get technical details and metadata for a specific event by ID."""
            return get_event_details(user_id=user_id, event_id=event_id)
            
        # Create agent instance for general chat with wrapped tools
        agent_tools = [
            list_cameras_wrapper, find_camera_wrapper, check_camera_health_wrapper,
            get_detections_wrapper, get_deployed_agents_wrapper, get_event_details_wrapper
        ]
        
        # Add dynamic time context
        from ....agents.utils.time_context import get_current_time_context
        def instruction_provider(context):
            return get_current_time_context()
            
        agent = create_general_chat_agent(tools=agent_tools, instruction=instruction_provider)
        
        self._sessions[session_id] = (adk_session, agent)
        self._session_user_map[session_id] = user_id

        runner = Runner(app_name=app_name, agent=agent, session_service=self.session_service)
        user_content = types.Content(role="user", parts=[types.Part(text=request.message)])

        yield {"event": "meta", "data": {"session_id": session_id}}

        emitted_so_far = ""
        final_response_text = ""
        last_model_response = ""

        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=adk_session.id,
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

                    delta = event_text
                    if emitted_so_far and event_text.startswith(emitted_so_far):
                        delta = event_text[len(emitted_so_far) :]

                    last_model_response = event_text
                    emitted_so_far = event_text

                    if delta:
                        yield {"event": "token", "data": {"delta": delta}}

            response_to_return = final_response_text.strip() if final_response_text.strip() else last_model_response.strip()
            if not response_to_return:
                response_to_return = "I apologize, but I didn't receive a proper response."
            response_to_return = _ensure_event_images_markdown(response_to_return)

            final_response = ChatMessageResponse(
                response=response_to_return,
                session_id=session_id,
                status="success",
            )
            yield {"event": "done", "data": final_response.model_dump()}
        except Exception as e:
            yield {"event": "error", "data": {"message": str(e)}}
            final_response = ChatMessageResponse(
                response=f"I encountered an error: {str(e)}. Please try again.",
                session_id=session_id or self._create_session_id(),
                status="error",
            )
            yield {"event": "done", "data": final_response.model_dump()}
    
    def _create_session_id(self) -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())

