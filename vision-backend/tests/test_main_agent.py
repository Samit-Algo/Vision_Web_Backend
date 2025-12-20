"""
Unit tests for main_agent.py
Includes functional tests and performance benchmarks
"""
import pytest
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
from pytz import UTC
from pathlib import Path

# Import the modules to test
from app.agents.main_agent import (
    get_current_time_context,
    build_instruction_dynamic_with_session,
    create_agent_for_session,
    static_instruction
)
from app.agents.session_state.agent_state import AgentState, reset_agent_state


class TestGetCurrentTimeContext:
    """Test cases for get_current_time_context function"""
    
    def test_returns_string(self):
        """Test that get_current_time_context returns a string"""
        result = get_current_time_context()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_contains_weekday(self):
        """Test that the result contains a weekday name"""
        result = get_current_time_context()
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        assert any(day in result for day in weekdays)
    
    def test_contains_date_format(self):
        """Test that the result contains date format"""
        result = get_current_time_context()
        # Should contain month name and year
        months = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
        assert any(month in result for month in months)
        assert "UTC" in result
    
    def test_performance(self):
        """Performance test: should be fast (< 10ms)"""
        start_time = time.perf_counter()
        for _ in range(100):
            get_current_time_context()
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
        assert avg_time < 10, f"get_current_time_context took {avg_time:.2f}ms on average"


class TestBuildInstructionDynamicWithSession:
    """Test cases for build_instruction_dynamic_with_session function"""
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock ReadonlyContext"""
        context = Mock()
        context.state = {}
        return context
    
    @pytest.fixture
    def session_id(self):
        """Test session ID"""
        return "test_session_123"
    
    def setup_method(self):
        """Reset agent state before each test"""
        reset_agent_state("test_session_123")
    
    def test_instruction_for_no_state(self, mock_context, session_id):
        """Test instruction generation when no agent state exists"""
        instruction = build_instruction_dynamic_with_session(mock_context, session_id)
        
        assert isinstance(instruction, str)
        assert len(instruction) > 0
        assert "No agent state initialized yet" in instruction
        assert "KNOWLEDGE BASE" in instruction
        assert "CURRENT AGENT STATE" in instruction
    
    def test_instruction_with_initialized_state(self, mock_context, session_id):
        """Test instruction generation when agent state is initialized"""
        from app.agents.session_state.agent_state import get_agent_state
        
        # Initialize state
        agent_state = get_agent_state(session_id)
        agent_state.rule_id = "class_presence"
        agent_state.status = "COLLECTING"
        agent_state.fields = {"camera_id": "CAM-001", "class": "Van"}
        agent_state.missing_fields = ["zone"]
        
        instruction = build_instruction_dynamic_with_session(mock_context, session_id)
        
        assert isinstance(instruction, str)
        assert "rule_id: class_presence" in instruction
        assert "status: COLLECTING" in instruction
        assert "CAM-001" in instruction
        assert "missing_fields" in instruction
    
    def test_instruction_contains_knowledge_base(self, mock_context, session_id):
        """Test that instruction contains knowledge base rules"""
        instruction = build_instruction_dynamic_with_session(mock_context, session_id)
        
        # Should contain rule structure (at minimum, the rules array should be mentioned)
        assert "rules" in instruction.lower() or "KNOWLEDGE BASE" in instruction
    
    def test_instruction_contains_time_context(self, mock_context, session_id):
        """Test that instruction contains current time context"""
        instruction = build_instruction_dynamic_with_session(mock_context, session_id)
        
        assert "Current Date and Time" in instruction or "CURRENT TIME CONTEXT" in instruction
    
    def test_instruction_for_confirmation_state(self, mock_context, session_id):
        """Test instruction generation in CONFIRMATION state"""
        from app.agents.session_state.agent_state import get_agent_state
        
        agent_state = get_agent_state(session_id)
        agent_state.rule_id = "class_presence"
        agent_state.status = "CONFIRMATION"
        agent_state.fields = {"camera_id": "CAM-001", "class": "Van", "zone": {"type": "polygon"}}
        agent_state.missing_fields = []
        
        instruction = build_instruction_dynamic_with_session(mock_context, session_id)
        
        assert "status: CONFIRMATION" in instruction
        assert "missing_fields" in instruction or "[]" in instruction
    
    def test_performance(self, mock_context, session_id):
        """Performance test: should be reasonably fast (< 100ms)"""
        start_time = time.perf_counter()
        for _ in range(10):
            build_instruction_dynamic_with_session(mock_context, session_id)
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 10 * 1000  # Convert to ms
        assert avg_time < 100, f"build_instruction_dynamic_with_session took {avg_time:.2f}ms on average"


class TestCreateAgentForSession:
    """Test cases for create_agent_for_session function"""
    
    @pytest.fixture
    def mock_groq_key(self):
        """Mock GROQ_API_KEY environment variable"""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key_12345"}):
            yield
    
    @pytest.fixture
    def mock_kb_data(self):
        """Mock knowledge base data"""
        return {
            "rules": [
                {
                    "rule_id": "class_presence",
                    "rule_name": "Class Presence",
                    "model": "yolov8",
                    "required_fields_from_user": ["camera_id", "class"],
                    "defaults": {"confidence": 0.5, "fps": 30}
                }
            ]
        }
    
    @pytest.fixture
    def mock_open_kb_file(self, mock_kb_data):
        """Mock knowledge base file reading"""
        kb_path = Path(__file__).resolve().parent.parent / "app" / "knowledge_base" / "vision_rule_knowledge_base.json"
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_kb_data))):
            with patch("app.agents.main_agent.KB_PATH", kb_path):
                yield
    
    def setup_method(self):
        """Setup before each test"""
        # Clear any existing agent states
        reset_agent_state("default")
        reset_agent_state("test_session")
    
    @patch('app.agents.main_agent.LlmAgent')
    def test_creates_agent_with_valid_key(self, mock_llm_agent_class, mock_groq_key, mock_open_kb_file):
        """Test that agent is created successfully with valid API key"""
        mock_llm_agent_instance = Mock()
        mock_llm_agent_class.return_value = mock_llm_agent_instance
        
        agent = create_agent_for_session("test_session", "user_123")
        
        assert agent is not None
        mock_llm_agent_class.assert_called_once()
        
        # Verify call arguments
        call_kwargs = mock_llm_agent_class.call_args[1]
        assert call_kwargs["name"] == "main_agent"
        assert call_kwargs["model"] == "groq/llama-3.3-70b-versatile"
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 3  # initialize_state, set_field_value, save_to_db
    
    @patch.dict(os.environ, {}, clear=True)
    def test_raises_error_without_api_key(self, mock_open_kb_file):
        """Test that ValueError is raised when GROQ_API_KEY is not set"""
        with pytest.raises(ValueError) as exc_info:
            create_agent_for_session("test_session")
        
        assert "GROQ_API_KEY" in str(exc_info.value)
    
    @patch('app.agents.main_agent.LlmAgent')
    @patch('app.agents.main_agent.FunctionTool')
    def test_agent_has_correct_tools(self, mock_function_tool_class, mock_llm_agent_class, 
                                     mock_groq_key, mock_open_kb_file):
        """Test that agent is created with correct tools"""
        mock_llm_agent_instance = Mock()
        mock_llm_agent_class.return_value = mock_llm_agent_instance
        
        create_agent_for_session("test_session", "user_123")
        
        # Verify FunctionTool was called 3 times (once for each tool)
        assert mock_function_tool_class.call_count == 3
    
    @patch('app.agents.main_agent.LlmAgent')
    def test_different_sessions_create_different_agents(self, mock_llm_agent_class, 
                                                        mock_groq_key, mock_open_kb_file):
        """Test that different sessions can create separate agent instances"""
        mock_llm_agent_instance_1 = Mock()
        mock_llm_agent_instance_2 = Mock()
        mock_llm_agent_class.side_effect = [mock_llm_agent_instance_1, mock_llm_agent_instance_2]
        
        agent1 = create_agent_for_session("session_1", "user_1")
        agent2 = create_agent_for_session("session_2", "user_2")
        
        assert agent1 != agent2
        assert mock_llm_agent_class.call_count == 2
    
    @patch('app.agents.main_agent.LlmAgent')
    def test_agent_has_static_and_dynamic_instructions(self, mock_llm_agent_class, 
                                                        mock_groq_key, mock_open_kb_file):
        """Test that agent has both static and dynamic instructions"""
        mock_llm_agent_instance = Mock()
        mock_llm_agent_class.return_value = mock_llm_agent_instance
        
        create_agent_for_session("test_session")
        
        call_kwargs = mock_llm_agent_class.call_args[1]
        assert "static_instruction" in call_kwargs
        assert "instruction" in call_kwargs
        assert call_kwargs["static_instruction"] == static_instruction
        assert callable(call_kwargs["instruction"])
    
    @patch('app.agents.main_agent.LlmAgent')
    def test_performance(self, mock_llm_agent_class, mock_groq_key, mock_open_kb_file):
        """Performance test: agent creation should be reasonably fast"""
        mock_llm_agent_instance = Mock()
        mock_llm_agent_class.return_value = mock_llm_agent_instance
        
        start_time = time.perf_counter()
        for i in range(5):
            create_agent_for_session(f"session_{i}")
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 5 * 1000  # Convert to ms
        assert avg_time < 500, f"create_agent_for_session took {avg_time:.2f}ms on average"


class TestStaticInstruction:
    """Test cases for static_instruction constant"""
    
    def test_static_instruction_is_string(self):
        """Test that static_instruction is a string"""
        assert isinstance(static_instruction, str)
        assert len(static_instruction) > 0
    
    def test_contains_absolute_rules(self):
        """Test that static_instruction contains absolute rules"""
        assert "ABSOLUTE RULES" in static_instruction or "NEVER VIOLATE" in static_instruction
    
    def test_contains_question_rules(self):
        """Test that static_instruction contains question rules"""
        assert "QUESTION RULES" in static_instruction or "missing_fields" in static_instruction.lower()


class TestPerformanceBenchmarks:
    """Performance benchmarks for agent operations"""
    
    @pytest.fixture
    def mock_groq_key(self):
        """Mock GROQ_API_KEY environment variable"""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key_12345"}):
            yield
    
    @pytest.fixture
    def mock_kb_data(self):
        """Mock knowledge base data"""
        return {
            "rules": [
                {
                    "rule_id": "class_presence",
                    "rule_name": "Class Presence",
                    "model": "yolov8",
                    "required_fields_from_user": ["camera_id", "class"],
                    "defaults": {"confidence": 0.5, "fps": 30}
                }
            ]
        }
    
    @pytest.fixture
    def mock_open_kb_file(self, mock_kb_data):
        """Mock knowledge base file reading"""
        kb_path = Path(__file__).resolve().parent.parent / "app" / "knowledge_base" / "vision_rule_knowledge_base.json"
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_kb_data))):
            with patch("app.agents.main_agent.KB_PATH", kb_path):
                yield
    
    @patch('app.agents.main_agent.LlmAgent')
    def test_concurrent_agent_creation_performance(self, mock_llm_agent_class, 
                                                   mock_groq_key, mock_open_kb_file):
        """Test performance of creating multiple agents concurrently (simulated)"""
        mock_llm_agent_instance = Mock()
        mock_llm_agent_class.return_value = mock_llm_agent_instance
        
        num_agents = 10
        start_time = time.perf_counter()
        
        agents = []
        for i in range(num_agents):
            agents.append(create_agent_for_session(f"session_{i}"))
        
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000  # Convert to ms
        
        assert len(agents) == num_agents
        assert total_time < 5000, f"Creating {num_agents} agents took {total_time:.2f}ms"
        
        avg_time = total_time / num_agents
        print(f"\nAverage time per agent creation: {avg_time:.2f}ms")
    
    def test_instruction_building_performance(self, mock_open_kb_file):
        """Test performance of building instructions multiple times"""
        from app.agents.main_agent import build_instruction_dynamic_with_session
        from unittest.mock import Mock
        
        mock_context = Mock()
        mock_context.state = {}
        session_id = "perf_test_session"
        
        # Warm up
        build_instruction_dynamic_with_session(mock_context, session_id)
        
        num_iterations = 50
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            build_instruction_dynamic_with_session(mock_context, session_id)
        
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000  # Convert to ms
        
        assert total_time < 2000, f"Building {num_iterations} instructions took {total_time:.2f}ms"
        
        avg_time = total_time / num_iterations
        print(f"\nAverage time per instruction build: {avg_time:.2f}ms")


# Run performance tests with: pytest -v -s tests/test_main_agent.py::TestPerformanceBenchmarks

