# Unit Tests for Agents Module

## Overview

This test suite provides comprehensive unit tests for the `app/agents/main_agent.py` module, including functional tests and performance benchmarks.

## Test Structure

### TestGetCurrentTimeContext
Tests for the `get_current_time_context()` function:
- Validates return type and format
- Checks for weekday, date, and time components
- Performance benchmark (< 10ms for 100 calls)

### TestBuildInstructionDynamicWithSession
Tests for the `build_instruction_dynamic_with_session()` function:
- Instruction generation for different agent states (no state, initialized, confirmation)
- Knowledge base inclusion
- Time context inclusion
- Performance benchmark (< 100ms per instruction)

### TestCreateAgentForSession
Tests for the `create_agent_for_session()` function:
- Agent creation with valid API key
- Error handling for missing API key
- Tool registration (3 tools: initialize_state, set_field_value, save_to_db)
- Session isolation
- Static and dynamic instruction assignment
- Performance benchmark (< 500ms per agent creation)

### TestStaticInstruction
Tests for the static instruction constant:
- Validates structure and content
- Checks for critical rule sections

### TestPerformanceBenchmarks
Performance benchmarks:
- Concurrent agent creation (10 agents)
- Instruction building performance (50 iterations)

## Running Tests

### Run all tests:
```bash
pytest tests/test_main_agent.py -v
```

### Run with performance output:
```bash
pytest tests/test_main_agent.py::TestPerformanceBenchmarks -v -s
```

### Run with timing information:
```bash
pytest tests/test_main_agent.py -v --durations=10
```

### Run specific test class:
```bash
pytest tests/test_main_agent.py::TestGetCurrentTimeContext -v
```

## Test Results Summary

**All 21 tests pass successfully.**

### Performance Metrics:
- **Agent Creation**: ~0.06ms average per agent
- **Instruction Building**: ~0.22ms average per instruction
- **Time Context Generation**: < 10ms for 100 calls

All performance benchmarks meet their respective thresholds.

## Conversation Style Tests

For testing actual conversation flows and agent creation scenarios, see:
- `test_agent_conversation_scenarios.py` - Comprehensive conversation scenario tests (tool-level)
- `test_llm_conversation_workflow.py` - Full LLM-powered conversation workflow tests
- `CONVERSATION_SCENARIOS.md` - Detailed documentation of all scenarios

### Quick Start for Conversation Tests

```bash
# Run all conversation scenario tests (tool-level, no LLM required)
pytest tests/test_agent_conversation_scenarios.py -v

# Run LLM-powered conversation workflow tests (requires GROQ_API_KEY)
pytest tests/test_llm_conversation_workflow.py -v -s

# 17 tool-level tests + 9 LLM workflow tests covering:
# - 10 different conversation scenarios
# - 4 edge case error scenarios
# - 3 state transition tests
# - 6 full LLM conversation workflows
# - 3 LLM state management tests
```

### LLM Conversation Workflow Tests

The LLM conversation workflow tests (`test_llm_conversation_workflow.py`) use the actual LLM agent to test end-to-end conversations. These tests:

- **Require GROQ_API_KEY**: Set environment variable before running
- **Test real LLM interactions**: Validate agent interprets user messages correctly
- **Verify tool calls**: Ensure agent calls tools (initialize_state, set_field_value, save_to_db) appropriately
- **Check state management**: Validate state transitions through conversation flow
- **Test multiple scenarios**: Simple creation, gradual gathering, zone requirements, corrections, etc.

**Note**: LLM tests are skipped automatically if GROQ_API_KEY is not set.

The tool-level conversation tests (`test_agent_conversation_scenarios.py`) simulate real user interactions by calling the agent tools in sequence, validating state transitions, and ensuring correct behavior across multiple rule types and use cases without requiring LLM calls.

## Dependencies

- pytest >= 7.0.0
- pytest-asyncio >= 0.21.0

## Mocking Strategy

The tests use comprehensive mocking for:
- `LlmAgent` class (Google ADK)
- `FunctionTool` class
- Knowledge base file reading
- Environment variables (GROQ_API_KEY)
- Session state management

This ensures tests are isolated and don't require actual API connections or database access.

