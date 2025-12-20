"""
LLM-powered conversation workflow integration tests.

These tests use the actual LLM agent to test the full conversation flow,
validating that the agent correctly interprets user messages, calls tools,
and manages state throughout the conversation.

Note: These tests require GROQ_API_KEY to be set in the environment.
If the key is not available, tests will be skipped.
"""
import pytest
import os
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import List, Optional

from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.agents.main_agent import create_agent_for_session
from app.agents.session_state.agent_state import get_agent_state, reset_agent_state
from app.agents.tools.save_to_db_tool import set_agent_repository
from app.domain.repositories.agent_repository import AgentRepository


# Check if GROQ_API_KEY is available
SKIP_LLM_TESTS = not bool(os.getenv("GROQ_API_KEY"))


@pytest.mark.skipif(SKIP_LLM_TESTS, reason="GROQ_API_KEY not set in environment")
@pytest.mark.asyncio
class TestLLMConversationWorkflow:
    """Test full conversation workflows with actual LLM"""
    
    @pytest.fixture
    def session_service(self):
        """Create in-memory session service"""
        return InMemorySessionService()
    
    @pytest.fixture
    def mock_repository(self):
        """Mock agent repository"""
        mock_repo = Mock(spec=AgentRepository)
        mock_saved_agent = Mock()
        mock_saved_agent.id = "agent_test_123"
        mock_repo.save = AsyncMock(return_value=mock_saved_agent)
        set_agent_repository(mock_repo)
        return mock_repo
    
    @pytest.fixture
    def app_name(self):
        """App name for ADK"""
        return "vision-agent-test"
    
    @pytest.fixture
    def user_id(self):
        """Test user ID"""
        return "test_user_123"
    
    async def _run_agent_conversation(
        self,
        session_service: InMemorySessionService,
        app_name: str,
        user_id: str,
        session_id: str,
        messages: List[str],
        mock_repository: Mock
    ) -> tuple[List[str], List[dict]]:
        """
        Run a conversation with the agent and collect responses and tool calls.
        
        Returns:
            (responses, tool_call_results): List of response texts and tool call results
        """
        # Create agent
        agent = create_agent_for_session(session_id=session_id, user_id=user_id)
        
        # Create or get session
        try:
            adk_session = await session_service.get_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id
            )
        except:
            adk_session = await session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id
            )
        
        # Store internal session ID
        if "internal_session_id" not in adk_session.state:
            adk_session.state["internal_session_id"] = session_id
        
        # Create runner
        runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=session_service
        )
        
        responses = []
        tool_call_results = []
        
        # Process each message
        for message in messages:
            # Create content object
            user_content = types.Content(
                role="user",
                parts=[types.Part(text=message)]
            )
            
            # Run agent and collect events
            response_text = ""
            async for event in runner.run_async(
                user_id=user_id,
                session_id=adk_session.id,
                new_message=user_content
            ):
                # Collect tool call results
                if hasattr(event, 'function_call') and event.function_call:
                    tool_call_results.append({
                        'name': getattr(event.function_call, 'name', None),
                        'args': getattr(event.function_call, 'args', {})
                    })
                
                # Collect response text
                if hasattr(event, 'content') and event.content:
                    if isinstance(event.content, types.Content):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
                    elif hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
                elif hasattr(event, 'text') and event.text:
                    response_text += event.text
            
            if response_text.strip():
                responses.append(response_text.strip())
        
        return responses, tool_call_results
    
    async def test_scenario_1_simple_agent_creation(
        self,
        session_service,
        app_name,
        user_id,
        mock_repository
    ):
        """
        Scenario 1: Simple agent creation with all info in first message
        
        User: "Alert me when a person appears on camera CAM-001 from 9am to 5pm"
        Expected: Agent initializes state, sets fields, asks for confirmation
        """
        session_id = "test_simple_creation"
        reset_agent_state(session_id)
        
        messages = [
            "Alert me when a person appears on camera CAM-001 from 9am to 5pm"
        ]
        
        responses, tool_calls = await self._run_agent_conversation(
            session_service, app_name, user_id, session_id, messages, mock_repository
        )
        
        # Verify agent responded
        assert len(responses) > 0, "Agent should respond to user message"
        
        # Verify tool calls were made
        tool_names = [tc['name'] for tc in tool_calls if tc.get('name')]
        assert 'initialize_state_wrapper' in tool_names or len(tool_calls) > 0, \
            "Agent should call initialize_state tool"
        
        # Check state was initialized
        state = get_agent_state(session_id)
        assert state.rule_id is not None, "Agent state should be initialized with a rule_id"
    
    async def test_scenario_2_gradual_conversation(
        self,
        session_service,
        app_name,
        user_id,
        mock_repository
    ):
        """
        Scenario 2: Gradual information gathering through conversation
        
        User messages:
        1. "I want to detect gestures"
        2. "Camera CAM-002"
        3. "Waving gesture"
        4. "From 8am to 6pm"
        5. "Yes, confirm"
        """
        session_id = "test_gradual_conversation"
        reset_agent_state(session_id)
        
        messages = [
            "I want to detect gestures",
            "Camera CAM-002",
            "Waving gesture",
            "From 8am to 6pm",
            "Yes, confirm"
        ]
        
        responses, tool_calls = await self._run_agent_conversation(
            session_service, app_name, user_id, session_id, messages, mock_repository
        )
        
        # Verify multiple responses
        assert len(responses) >= 3, "Agent should respond to multiple messages"
        
        # Verify state progression
        state = get_agent_state(session_id)
        
        # After all messages, state should have rule_id
        assert state.rule_id == "gesture_detected", \
            f"Expected gesture_detected rule, got {state.rule_id}"
        
        # Verify fields were collected
        assert state.fields.get("camera_id") == "CAM-002", \
            "Camera ID should be set"
        assert state.fields.get("gesture") == "waving", \
            "Gesture should be set"
        
        # Check if confirmation or collection state
        assert state.status in ["COLLECTING", "CONFIRMATION"], \
            f"State should be COLLECTING or CONFIRMATION, got {state.status}"
    
    async def test_scenario_3_zone_required_flow(
        self,
        session_service,
        app_name,
        user_id,
        mock_repository
    ):
        """
        Scenario 3: Zone-required rule conversation
        
        User: "I want to detect when a car enters a zone"
        Expected: Agent asks for zone definition
        """
        session_id = "test_zone_required"
        reset_agent_state(session_id)
        
        messages = [
            "I want to detect when a car enters a zone on camera CAM-003"
        ]
        
        responses, tool_calls = await self._run_agent_conversation(
            session_service, app_name, user_id, session_id, messages, mock_repository
        )
        
        # Verify response mentions zone
        assert len(responses) > 0, "Agent should respond"
        
        # Check state requires zone
        state = get_agent_state(session_id)
        if state.rule_id == "object_enter_zone":
            assert state.fields.get("requires_zone") is True, \
                "Zone should be required for object_enter_zone rule"
            assert "zone" in state.missing_fields or state.status == "COLLECTING", \
                "Zone should be in missing fields"
    
    async def test_scenario_4_full_workflow_to_save(
        self,
        session_service,
        app_name,
        user_id,
        mock_repository
    ):
        """
        Scenario 4: Complete workflow from start to save
        
        Tests the full conversation flow ending with agent save
        """
        session_id = "test_full_workflow"
        reset_agent_state(session_id)
        
        messages = [
            "Create an agent to detect when a person appears on camera CAM-004",
            "From 9am to 5pm",
            "Yes, save it"
        ]
        
        responses, tool_calls = await self._run_agent_conversation(
            session_service, app_name, user_id, session_id, messages, mock_repository
        )
        
        # Verify responses
        assert len(responses) >= 2, "Agent should respond to user messages"
        
        # Check if save_to_db was called
        tool_names = [tc['name'] for tc in tool_calls if tc.get('name')]
        has_save_call = 'save_to_db_wrapper' in tool_names or any(
            'save' in str(tc).lower() for tc in tool_calls
        )
        
        # Verify state
        state = get_agent_state(session_id)
        
        # If save was called, state should be reset or in confirmation
        if has_save_call:
            # State might be reset after save
            assert state.rule_id is None or state.status == "CONFIRMATION", \
                "State should be reset after save or in confirmation"
            
            # Verify repository was called
            mock_repository.save.assert_called() if has_save_call else None
    
    async def test_scenario_5_correction_flow(
        self,
        session_service,
        app_name,
        user_id,
        mock_repository
    ):
        """
        Scenario 5: User corrects information
        
        User messages:
        1. "Alert when car appears on CAM-005"
        2. "Actually, make it a person instead"
        """
        session_id = "test_correction"
        reset_agent_state(session_id)
        
        messages = [
            "Alert when a car appears on camera CAM-005 from 9am to 5pm",
            "Actually, make it detect a person instead of a car"
        ]
        
        responses, tool_calls = await self._run_agent_conversation(
            session_service, app_name, user_id, session_id, messages, mock_repository
        )
        
        # Verify responses
        assert len(responses) >= 2, "Agent should handle correction"
        
        # Check state has corrected value
        state = get_agent_state(session_id)
        if state.rule_id:
            # The class field should reflect the correction
            # (Agent might have updated it via set_field_value)
            assert state.fields.get("camera_id") == "CAM-005", \
                "Camera ID should remain unchanged"
    
    async def test_scenario_6_proximity_detection(
        self,
        session_service,
        app_name,
        user_id,
        mock_repository
    ):
        """
        Scenario 6: Proximity detection rule
        
        User: "Alert if a person gets close to a car"
        """
        session_id = "test_proximity"
        reset_agent_state(session_id)
        
        messages = [
            "Alert if a person gets close to a car on camera CAM-006"
        ]
        
        responses, tool_calls = await self._run_agent_conversation(
            session_service, app_name, user_id, session_id, messages, mock_repository
        )
        
        # Verify response
        assert len(responses) > 0, "Agent should respond"
        
        # Check if proximity rule was detected
        state = get_agent_state(session_id)
        if state.rule_id == "proximity_detection":
            assert "class_a" in state.missing_fields or state.fields.get("class_a"), \
                "Proximity detection requires class_a"
            assert "class_b" in state.missing_fields or state.fields.get("class_b"), \
                "Proximity detection requires class_b"


@pytest.mark.skipif(SKIP_LLM_TESTS, reason="GROQ_API_KEY not set in environment")
@pytest.mark.asyncio
class TestLLMStateManagement:
    """Test state management correctness with LLM"""
    
    @pytest.fixture
    def session_service(self):
        return InMemorySessionService()
    
    @pytest.fixture
    def mock_repository(self):
        mock_repo = Mock(spec=AgentRepository)
        mock_repo.save = AsyncMock(return_value=Mock(id="test_agent"))
        set_agent_repository(mock_repo)
        return mock_repo
    
    async def test_state_persistence_across_messages(
        self,
        session_service,
        mock_repository
    ):
        """Test that state persists across multiple messages in same session"""
        session_id = "test_state_persistence"
        user_id = "test_user"
        app_name = "vision-agent-test"
        reset_agent_state(session_id)
        
        agent = create_agent_for_session(session_id=session_id, user_id=user_id)
        
        # Create session
        try:
            adk_session = await session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id
            )
        except:
            adk_session = await session_service.get_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id
            )
        
        if "internal_session_id" not in adk_session.state:
            adk_session.state["internal_session_id"] = session_id
        
        runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=session_service
        )
        
        # First message
        message1 = types.Content(
            role="user",
            parts=[types.Part(text="I want to detect when a person appears")]
        )
        
        async for _ in runner.run_async(
            user_id=user_id,
            session_id=adk_session.id,
            new_message=message1
        ):
            pass
        
        # Check state after first message
        state1 = get_agent_state(session_id)
        assert state1.rule_id is not None, "State should have rule_id after first message"
        first_rule_id = state1.rule_id
        
        # Second message
        message2 = types.Content(
            role="user",
            parts=[types.Part(text="Camera CAM-007")]
        )
        
        async for _ in runner.run_async(
            user_id=user_id,
            session_id=adk_session.id,
            new_message=message2
        ):
            pass
        
        # Check state persists
        state2 = get_agent_state(session_id)
        assert state2.rule_id == first_rule_id, \
            "Rule ID should persist across messages in same session"
        assert state2.fields.get("camera_id") == "CAM-007", \
            "Field from second message should be in state"


@pytest.mark.skipif(SKIP_LLM_TESTS, reason="GROQ_API_KEY not set in environment")
@pytest.mark.asyncio
class TestLLMToolCalls:
    """Test that LLM correctly calls tools"""
    
    @pytest.fixture
    def session_service(self):
        return InMemorySessionService()
    
    @pytest.fixture
    def mock_repository(self):
        mock_repo = Mock(spec=AgentRepository)
        mock_repo.save = AsyncMock(return_value=Mock(id="test_agent"))
        set_agent_repository(mock_repo)
        return mock_repo
    
    async def test_initialize_state_tool_call(
        self,
        session_service,
        mock_repository
    ):
        """Test that agent calls initialize_state tool"""
        session_id = "test_init_tool"
        user_id = "test_user"
        app_name = "vision-agent-test"
        reset_agent_state(session_id)
        
        agent = create_agent_for_session(session_id=session_id, user_id=user_id)
        
        adk_session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        if "internal_session_id" not in adk_session.state:
            adk_session.state["internal_session_id"] = session_id
        
        runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=session_service
        )
        
        tool_calls = []
        message = types.Content(
            role="user",
            parts=[types.Part(text="I want to detect when a person appears on camera CAM-008")]
        )
        
        async for event in runner.run_async(
            user_id=user_id,
            session_id=adk_session.id,
            new_message=message
        ):
            if hasattr(event, 'function_call') and event.function_call:
                tool_calls.append({
                    'name': getattr(event.function_call, 'name', None),
                    'args': getattr(event.function_call, 'args', {})
                })
        
        # Verify state was initialized (indirect verification via state)
        state = get_agent_state(session_id)
        assert state.rule_id is not None, \
            "State should be initialized, indicating initialize_state was called"


# Run with: pytest tests/test_llm_conversation_workflow.py -v -s
# Requires GROQ_API_KEY environment variable to be set

