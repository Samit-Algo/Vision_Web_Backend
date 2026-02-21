"""
Agent state schema for the vision-rule flow.
Backend (tools) update these fields; LLM only triggers tools.
Keys that can be updated by multiple tool calls in one step use a reducer (last-write-wins).
"""

from typing import Annotated, Any, NotRequired

from langchain.agents import AgentState


def last_wins_reducer(previous: Any, current: Any) -> Any:
    """Reducer: when multiple updates in one step, use the last (current) value."""
    return current if current is not None else previous


class VisionRuleState(AgentState):
    """State for the vision-rule agent. Extended by tools only."""

    rule_id: NotRequired[Annotated[str | None, last_wins_reducer]]
    config: NotRequired[Annotated[dict[str, Any], last_wins_reducer]]
    missing_fields: NotRequired[Annotated[list[str], last_wins_reducer]]
    status: NotRequired[Annotated[str, last_wins_reducer]]
    user_id: NotRequired[Annotated[str | None, last_wins_reducer]]
    session_id: NotRequired[Annotated[str | None, last_wins_reducer]]
