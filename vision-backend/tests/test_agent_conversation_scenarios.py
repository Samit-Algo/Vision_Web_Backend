"""
Conversation-style integration tests for agent creation scenarios.

These tests simulate real user conversations and validate:
- Conversation flow and state transitions
- Tool call sequences
- State management correctness
- Multiple rule types and scenarios
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path

from app.agents.session_state.agent_state import (
    AgentState,
    get_agent_state,
    reset_agent_state,
    set_agent_state
)
from app.agents.tools.initialize_state_tool import initialize_state
from app.agents.tools.set_field_value_tool import set_field_value
from app.agents.tools.save_to_db_tool import save_to_db, set_agent_repository


class TestConversationScenarios:
    """Test conversation flows for agent creation"""
    
    @pytest.fixture
    def mock_repository(self):
        """Mock agent repository for save operations"""
        mock_repo = Mock()
        mock_saved_agent = Mock()
        mock_saved_agent.id = "agent_123"
        mock_repo.save = AsyncMock(return_value=mock_saved_agent)
        set_agent_repository(mock_repo)
        return mock_repo
    
    @pytest.fixture
    def session_id(self):
        """Test session ID"""
        return "test_conversation_session"
    
    def setup_method(self):
        """Reset state before each test"""
        reset_agent_state("test_conversation_session")
        reset_agent_state("scenario_1")
        reset_agent_state("scenario_2")
        reset_agent_state("scenario_3")
    
    def test_scenario_1_simple_class_presence_flow(self, session_id, mock_repository):
        """
        Scenario 1: Simple class presence detection
        User provides all information in first message, confirms, agent is saved.
        """
        # Step 1: User says "Alert me when a person appears on camera CAM-001"
        # Simulated tool call: initialize_state
        result1 = initialize_state("class_presence", session_id)
        assert result1["rule_id"] == "class_presence"
        assert result1["status"] == "COLLECTING"
        
        # Check state after initialization
        state1 = get_agent_state(session_id)
        assert state1.rule_id == "class_presence"
        assert "camera_id" in state1.missing_fields
        
        # Step 2: User provides camera_id and class
        # Simulated tool call: set_field_value
        field_values = {
            "camera_id": "CAM-001",
            "class": "person",
            "start_time": "2025-01-20T09:00:00Z",
            "end_time": "2025-01-20T17:00:00Z"
        }
        result2 = set_field_value(json.dumps(field_values), session_id)
        assert result2["status"] == "CONFIRMATION"
        assert len(result2["updated_fields"]) == 4
        
        # Check state after field setting
        state2 = get_agent_state(session_id)
        assert state2.status == "CONFIRMATION"
        assert state2.fields["camera_id"] == "CAM-001"
        assert state2.fields["class"] == "person"
        assert len(state2.missing_fields) == 0
        
        # Step 3: User confirms "Yes, that's correct"
        # Simulated tool call: save_to_db
        result3 = save_to_db(session_id, "user_123")
        assert result3["status"] == "DONE"
        assert result3["saved"] is True
        mock_repository.save.assert_called_once()
        
        # Check state is reset after save
        state3 = get_agent_state(session_id)
        assert state3.rule_id is None
        assert state3.status == "COLLECTING"
    
    def test_scenario_2_gradual_information_gathering(self, session_id, mock_repository):
        """
        Scenario 2: Gradual information gathering
        User provides information piece by piece through multiple messages.
        """
        # Step 1: User says "I want to detect gestures"
        result1 = initialize_state("gesture_detected", session_id)
        assert result1["rule_id"] == "gesture_detected"
        
        state1 = get_agent_state(session_id)
        assert "camera_id" in state1.missing_fields
        assert "gesture" in state1.missing_fields
        
        # Step 2: User provides camera only
        result2 = set_field_value(json.dumps({"camera_id": "CAM-002"}), session_id)
        assert result2["status"] == "COLLECTING"
        
        state2 = get_agent_state(session_id)
        assert state2.fields["camera_id"] == "CAM-002"
        assert "gesture" in state2.missing_fields
        assert "camera_id" not in state2.missing_fields
        
        # Step 3: User provides gesture
        result3 = set_field_value(json.dumps({"gesture": "waving"}), session_id)
        assert result3["status"] == "COLLECTING"  # Still collecting if times are missing
        
        state3 = get_agent_state(session_id)
        assert state3.fields["gesture"] == "waving"
        
        # Step 4: User provides times
        result4 = set_field_value(json.dumps({
            "start_time": "2025-01-20T08:00:00Z",
            "end_time": "2025-01-20T18:00:00Z"
        }), session_id)
        assert result4["status"] == "CONFIRMATION"
        
        state4 = get_agent_state(session_id)
        assert state4.status == "CONFIRMATION"
        assert len(state4.missing_fields) == 0
    
    def test_scenario_3_zone_required_rule(self, session_id, mock_repository):
        """
        Scenario 3: Rule that requires zone (object_enter_zone)
        User must provide zone information.
        """
        # Step 1: Initialize zone-required rule
        result1 = initialize_state("object_enter_zone", session_id)
        assert result1["rule_id"] == "object_enter_zone"
        
        state1 = get_agent_state(session_id)
        assert state1.fields["requires_zone"] is True
        assert "zone" in state1.missing_fields
        
        # Step 2: Provide camera and class
        result2 = set_field_value(json.dumps({
            "camera_id": "CAM-003",
            "class": "car"
        }), session_id)
        
        state2 = get_agent_state(session_id)
        assert "zone" in state2.missing_fields  # Zone still required
        
        # Step 3: Provide zone
        zone_data = {
            "type": "polygon",
            "coordinates": [[0, 0], [100, 0], [100, 100], [0, 100]]
        }
        result3 = set_field_value(json.dumps({
            "zone": zone_data,
            "start_time": "2025-01-20T09:00:00Z",
            "end_time": "2025-01-20T17:00:00Z"
        }), session_id)
        assert result3["status"] == "CONFIRMATION"
        
        state3 = get_agent_state(session_id)
        assert state3.fields["zone"] == zone_data
        assert "zone" not in state3.missing_fields
    
    def test_scenario_4_patrol_mode_requires_additional_fields(self, session_id, mock_repository):
        """
        Scenario 4: Patrol mode requires interval_minutes and check_duration_seconds
        """
        # Step 1: Initialize with patrol mode
        result1 = initialize_state("class_presence", session_id)
        result2 = set_field_value(json.dumps({
            "camera_id": "CAM-004",
            "class": "person",
            "run_mode": "patrol",
            "start_time": "2025-01-20T09:00:00Z",
            "end_time": "2025-01-20T17:00:00Z"
        }), session_id)
        
        state2 = get_agent_state(session_id)
        # Patrol mode requires interval_minutes, check_duration_seconds, and zone
        assert "interval_minutes" in state2.missing_fields
        assert "check_duration_seconds" in state2.missing_fields
        assert "zone" in state2.missing_fields
        
        # Step 3: Provide patrol-specific fields
        result3 = set_field_value(json.dumps({
            "interval_minutes": 15,
            "check_duration_seconds": 30,
            "zone": {"type": "polygon", "coordinates": [[0, 0], [100, 0], [100, 100], [0, 100]]}
        }), session_id)
        
        state3 = get_agent_state(session_id)
        assert state3.status == "CONFIRMATION"
        assert len(state3.missing_fields) == 0
    
    def test_scenario_5_proximity_detection_rule(self, session_id, mock_repository):
        """
        Scenario 5: Proximity detection requires class_a, class_b, and distance
        """
        result1 = initialize_state("proximity_detection", session_id)
        
        state1 = get_agent_state(session_id)
        assert "class_a" in state1.missing_fields
        assert "class_b" in state1.missing_fields
        assert "distance" in state1.missing_fields
        
        # Provide all required fields
        result2 = set_field_value(json.dumps({
            "camera_id": "CAM-005",
            "class_a": "person",
            "class_b": "car",
            "distance": 5.0,
            "start_time": "2025-01-20T09:00:00Z",
            "end_time": "2025-01-20T17:00:00Z"
        }), session_id)
        
        state2 = get_agent_state(session_id)
        assert state2.status == "CONFIRMATION"
        assert state2.fields["class_a"] == "person"
        assert state2.fields["class_b"] == "car"
        assert state2.fields["distance"] == 5.0
    
    def test_scenario_6_correction_flow(self, session_id, mock_repository):
        """
        Scenario 6: User corrects information after confirmation
        """
        # Initialize and fill fields
        initialize_state("class_presence", session_id)
        set_field_value(json.dumps({
            "camera_id": "CAM-006",
            "class": "person",
            "start_time": "2025-01-20T09:00:00Z",
            "end_time": "2025-01-20T17:00:00Z"
        }), session_id)
        
        state1 = get_agent_state(session_id)
        assert state1.status == "CONFIRMATION"
        assert state1.fields["class"] == "person"
        
        # User corrects class
        result2 = set_field_value(json.dumps({"class": "car"}), session_id)
        
        state2 = get_agent_state(session_id)
        assert state2.fields["class"] == "car"
        assert state2.status == "CONFIRMATION"  # Back to confirmation
    
    def test_scenario_7_all_fields_in_first_message(self, session_id, mock_repository):
        """
        Scenario 7: User provides all information in the first message
        Tests that initialize_state + set_field_value works correctly together
        """
        # Simulate: User says "Alert me when a car appears on CAM-007 from 9am to 5pm"
        result1 = initialize_state("class_presence", session_id)
        
        # Immediately set all fields
        result2 = set_field_value(json.dumps({
            "camera_id": "CAM-007",
            "class": "car",
            "start_time": "2025-01-20T09:00:00Z",
            "end_time": "2025-01-20T17:00:00Z"
        }), session_id)
        
        state = get_agent_state(session_id)
        assert state.status == "CONFIRMATION"
        assert state.fields["camera_id"] == "CAM-007"
        assert state.fields["class"] == "car"
        assert len(state.missing_fields) == 0
    
    def test_scenario_8_defaults_applied_automatically(self, session_id):
        """
        Scenario 8: Verify that defaults from knowledge base are applied
        """
        result1 = initialize_state("class_presence", session_id)
        
        state = get_agent_state(session_id)
        # Check defaults are applied
        assert state.fields.get("model") == "YOLO-Safety-v1"  # From knowledge base
        assert state.fields.get("fps") == 15  # From defaults
        assert state.fields.get("confidence") == 0.7  # From defaults
        assert state.fields.get("run_mode") == "continuous"  # Default
    
    def test_scenario_9_rules_array_construction(self, session_id):
        """
        Scenario 9: Verify rules array is constructed correctly
        """
        initialize_state("class_presence", session_id)
        set_field_value(json.dumps({
            "camera_id": "CAM-009",
            "class": "person"
        }), session_id)
        
        state = get_agent_state(session_id)
        assert "rules" in state.fields
        rules = state.fields["rules"]
        assert len(rules) == 1
        assert rules[0]["type"] == "class_presence"
        assert rules[0]["class"] == "person"
        assert "label" in rules[0]  # Auto-generated label
    
    def test_scenario_10_multiple_corrections(self, session_id, mock_repository):
        """
        Scenario 10: User makes multiple corrections
        """
        initialize_state("class_presence", session_id)
        
        # First set of values
        set_field_value(json.dumps({
            "camera_id": "CAM-010",
            "class": "person"
        }), session_id)
        
        state1 = get_agent_state(session_id)
        assert state1.fields["class"] == "person"
        
        # Correction 1: Change class
        set_field_value(json.dumps({"class": "car"}), session_id)
        state2 = get_agent_state(session_id)
        assert state2.fields["class"] == "car"
        
        # Correction 2: Change camera
        set_field_value(json.dumps({"camera_id": "CAM-011"}), session_id)
        state3 = get_agent_state(session_id)
        assert state3.fields["camera_id"] == "CAM-011"
        assert state3.fields["class"] == "car"  # Previous value preserved


class TestConversationEdgeCases:
    """Test edge cases and error scenarios"""
    
    @pytest.fixture
    def session_id(self):
        return "edge_case_session"
    
    def setup_method(self):
        reset_agent_state("edge_case_session")
    
    def test_set_fields_before_initialization(self, session_id):
        """Test that setting fields before initialization raises error"""
        with pytest.raises(ValueError, match="Cannot set fields before rule selection"):
            set_field_value(json.dumps({"camera_id": "CAM-001"}), session_id)
    
    def test_save_without_confirmation(self, session_id):
        """Test that saving without confirmation raises error"""
        initialize_state("class_presence", session_id)
        
        state = get_agent_state(session_id)
        assert state.status == "COLLECTING"
        
        with pytest.raises(ValueError, match="not in CONFIRMATION state"):
            save_to_db(session_id, "user_123")
    
    def test_save_with_missing_fields(self, session_id):
        """Test that saving with missing fields raises error"""
        initialize_state("class_presence", session_id)
        set_field_value(json.dumps({"camera_id": "CAM-001"}), session_id)
        
        state = get_agent_state(session_id)
        assert len(state.missing_fields) > 0
        
        # Try to force status to CONFIRMATION (shouldn't happen in real flow)
        state.status = "CONFIRMATION"
        set_agent_state(state, session_id)
        
        with pytest.raises(ValueError, match="missing fields"):
            save_to_db(session_id, "user_123")
    
    def test_save_without_user_id(self, session_id):
        """Test that saving without user_id raises error"""
        initialize_state("class_presence", session_id)
        set_field_value(json.dumps({
            "camera_id": "CAM-001",
            "class": "person",
            "start_time": "2025-01-20T09:00:00Z",
            "end_time": "2025-01-20T17:00:00Z"
        }), session_id)
        
        state = get_agent_state(session_id)
        assert state.status == "CONFIRMATION"
        
        with pytest.raises(ValueError, match="user_id is required"):
            save_to_db(session_id, None)


class TestConversationStateTransitions:
    """Test state transition correctness"""
    
    @pytest.fixture
    def session_id(self):
        return "state_transition_session"
    
    def setup_method(self):
        reset_agent_state("state_transition_session")
    
    def test_state_transition_collecting_to_confirmation(self, session_id):
        """Test transition from COLLECTING to CONFIRMATION when all fields provided"""
        initialize_state("class_presence", session_id)
        
        state1 = get_agent_state(session_id)
        assert state1.status == "COLLECTING"
        
        set_field_value(json.dumps({
            "camera_id": "CAM-001",
            "class": "person",
            "start_time": "2025-01-20T09:00:00Z",
            "end_time": "2025-01-20T17:00:00Z"
        }), session_id)
        
        state2 = get_agent_state(session_id)
        assert state2.status == "CONFIRMATION"
    
    def test_state_remains_collecting_with_missing_fields(self, session_id):
        """Test that state remains COLLECTING when fields are missing"""
        initialize_state("class_presence", session_id)
        
        # Provide only camera_id, missing class and times
        set_field_value(json.dumps({"camera_id": "CAM-001"}), session_id)
        
        state = get_agent_state(session_id)
        assert state.status == "COLLECTING"
        assert len(state.missing_fields) > 0
    
    def test_state_reset_after_save(self, session_id):
        """Test that state is reset after successful save"""
        mock_repo = Mock()
        mock_repo.save = AsyncMock(return_value=Mock(id="agent_123"))
        set_agent_repository(mock_repo)
        
        initialize_state("class_presence", session_id)
        set_field_value(json.dumps({
            "camera_id": "CAM-001",
            "class": "person",
            "start_time": "2025-01-20T09:00:00Z",
            "end_time": "2025-01-20T17:00:00Z"
        }), session_id)
        
        save_to_db(session_id, "user_123")
        
        state = get_agent_state(session_id)
        assert state.rule_id is None
        assert state.status == "COLLECTING"
        assert len(state.fields) == 0


# Run with: pytest tests/test_agent_conversation_scenarios.py -v

