"""Agent state definition.

Owner: Member 5 (feature/agent-core).
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State carried through the LangGraph graph."""

    messages: Annotated[list, add_messages]
    tool_results: list[dict[str, Any]]
    iteration: int


def initial_state(user_message: str) -> AgentState:
    """Return a fresh AgentState for a new user request."""
    return {
        "messages": [HumanMessage(content=user_message)],
        "tool_results": [],
        "iteration": 0,
    }