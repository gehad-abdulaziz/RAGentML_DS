"""Agent state definition.

Owner: Member 5 (feature/agent-core).

Member 6 (logger, CLI, prompts) and the tests depend on this shape: agree on it early and
announce any change to the team.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State carried through the LangGraph graph.

    messages:     conversation history (user, assistant, tool messages); use the add_messages reducer.
    tool_results: structured tool outputs collected so far, e.g.
                  [{"iteration": 1, "tool": "predict_performance", "input": {...}, "output": {...}}]
    iteration:    int, number of agent iterations used so far (0 at start, hard limit MAX_ITERATIONS).
    """

    messages: Annotated[list, add_messages]
    tool_results: list[dict[str, Any]]
    iteration: int


# TODO (Member 5): decide whether to add fields (e.g., "final_answer", "stop_reason":
# "answered" | "needs_user_input" | "max_iterations") and tell Member 6.
# TODO (Member 5): helper to create the initial state from a user message:
def initial_state(user_message: str) -> AgentState:
    """Return a fresh AgentState for a new user request.

    TODO: messages=[HumanMessage(user_message)], tool_results=[], iteration=0
    """
    raise NotImplementedError("TODO (Member 5)")