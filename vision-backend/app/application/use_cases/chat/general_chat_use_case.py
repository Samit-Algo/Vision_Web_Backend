"""Use case for general chat with a simple Google ADK agent."""
import json
import uuid
from typing import Optional, AsyncGenerator, Dict, Any
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types

from ...dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ....agents.general_chat_agent import create_general_chat_agent


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
        
        # Create agent instance for general chat
        agent = create_general_chat_agent()
        
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
            
            return ChatMessageResponse(
                response=response_to_return if response_to_return else "I apologize, but I didn't receive a proper response.",
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

        agent = create_general_chat_agent()
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

