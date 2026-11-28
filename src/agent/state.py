"""Agent state definition."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State carried through the LangGraph graph (one user turn)."""

    messages: Annotated[list, add_messages]   # history + this turn's messages
    tool_results: list[dict[str, Any]]        # one record per tool call of this turn
    iteration: int                            # tool-calling rounds used in this turn
    # --- identity / long-term memory (filled by the app layer) ---
    user_id: Any
    session_id: str
    user_name: str
    memory_context: str                       # text injected into the system prompt
    profile_features: dict[str, Any]          # saved ML features of this student
    # --- guardrails ---
    blocked: bool
    block_reason: str
    guard_events: list[str]


def initial_state(
    user_message: str,
    history: list[BaseMessage] | None = None,
    user_id: Any = "anonymous",
    session_id: str = "default",
    user_name: str = "",
    memory_context: str = "",
    profile_features: dict[str, Any] | None = None,
) -> AgentState:
    """Return the AgentState for a new user turn (history comes before the new message)."""
    return {
        "messages": [*(history or []), HumanMessage(content=user_message)],
        "tool_results": [],
        "iteration": 0,
        "user_id": user_id,
        "session_id": session_id,
        "user_name": user_name,
        "memory_context": memory_context,
        "profile_features": dict(profile_features or {}),
        "blocked": False,
        "block_reason": "",
        "guard_events": [],
    }
