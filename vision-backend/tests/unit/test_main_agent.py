"""
Unit tests for main_agent.py
"""
import os
from unittest.mock import Mock, patch

import pytest

from app.agents.main_agent import (
    build_dynamic_instruction,
    create_agent_for_session,
    STATIC_INSTRUCTION,
)
from app.agents.session_state.agent_state import get_agent_state, reset_agent_state
from app.agents.utils.time_context import get_current_time_context, get_short_time_context


class TestGetCurrentTimeContext:
    """Tests for get_current_time_context (full format)"""

    def test_returns_string(self):
        result = get_current_time_context()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_weekday(self):
        result = get_current_time_context()
        weekdays = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]
        assert any(day in result for day in weekdays)

    def test_contains_date_components(self):
        result = get_current_time_context()
        assert "UTC" in result or "IST" in result
        assert "TIME_CONTEXT" in result or "NOW_" in result


class TestGetShortTimeContext:
    """Tests for get_short_time_context (minimal per-turn format)"""

    def test_returns_string(self):
        result = get_short_time_context()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_ist(self):
        result = get_short_time_context()
        assert "IST" in result or len(result) > 0


class TestBuildDynamicInstruction:
    """Tests for build_dynamic_instruction (per-turn minimal instruction)"""

    def setup_method(self):
        reset_agent_state("test_session_main")

    def test_instruction_for_uninitialized_state(self):
        instruction = build_dynamic_instruction("test_session_main")
        assert isinstance(instruction, str)
        assert len(instruction) > 0
        assert "UNINITIALIZED" in instruction
        assert "CURRENT_STATE" in instruction

    def test_instruction_with_collecting_state(self):
        state = get_agent_state("test_session_main")
        state.rule_id = "class_presence"
        state.status = "COLLECTING"
        state.fields = {"camera_id": "CAM-001", "class": "Van"}
        state.missing_fields = ["zone"]

        instruction = build_dynamic_instruction("test_session_main")
        assert "class_presence" in instruction
        assert "COLLECTING" in instruction
        assert "CAM-001" in instruction or "zone" in instruction

    def test_instruction_for_confirmation_state(self):
        state = get_agent_state("test_session_main")
        state.rule_id = "class_presence"
        state.status = "CONFIRMATION"
        state.fields = {"camera_id": "CAM-001", "class": "Van", "zone": {"type": "polygon"}}
        state.missing_fields = []

        instruction = build_dynamic_instruction("test_session_main")
        assert "CONFIRMATION" in instruction


class TestCreateAgentForSession:
    """Tests for create_agent_for_session"""

    @pytest.fixture
    def mock_groq_key(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key_12345"}):
            yield

    def setup_method(self):
        reset_agent_state("default")
        reset_agent_state("test_session")

    @patch("app.agents.main_agent.LlmAgent")
    def test_creates_agent_with_valid_key(self, mock_llm_class, mock_groq_key):
        mock_llm_class.return_value = Mock()
        agent = create_agent_for_session("test_session", "user_123")
        assert agent is not None
        mock_llm_class.assert_called_once()
        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs["name"] == "main_agent"
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) >= 3

    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                create_agent_for_session("test_session")

    @patch("app.agents.main_agent.LlmAgent")
    def test_agent_has_static_and_dynamic_instructions(
        self, mock_llm_class, mock_groq_key
    ):
        mock_llm_class.return_value = Mock()
        create_agent_for_session("test_session")
        call_kwargs = mock_llm_class.call_args[1]
        assert "static_instruction" in call_kwargs
        assert "instruction" in call_kwargs
        assert call_kwargs["static_instruction"] == STATIC_INSTRUCTION
        assert callable(call_kwargs["instruction"])


class TestStaticInstruction:
    """Tests for STATIC_INSTRUCTION (cached at startup, includes rules catalog)"""

    def test_is_non_empty_string(self):
        assert isinstance(STATIC_INSTRUCTION, str)
        assert len(STATIC_INSTRUCTION) > 0

    def test_contains_critical_rules(self):
        assert "NON-NEGOTIABLE" in STATIC_INSTRUCTION or "Never" in STATIC_INSTRUCTION
        assert "missing_fields" in STATIC_INSTRUCTION

    def test_contains_rules_catalog(self):
        assert "RULES_CATALOG_JSON" in STATIC_INSTRUCTION or "rule_id" in STATIC_INSTRUCTION
