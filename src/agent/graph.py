"""Builds the LangGraph graph.

Owner: Member 5 (feature/agent-core).

Flow: START -> llm_node -> (conditional edge) -> tool_node -> llm_node ... -> END
The loop stops when the LLM gives a final answer / asks the user for missing info, or when
the iteration counter reaches MAX_ITERATIONS (5).
"""
from __future__ import annotations

from src.agent.state import AgentState
from src.config import MAX_ITERATIONS  # noqa: F401  (used by the TODOs below)


def should_continue(state: AgentState) -> str:
    """Conditional edge after llm_node.

    Returns "tools" to continue to tool_node, or "end" to finish.

    TODO:
      - if the last AI message has tool calls AND iteration < MAX_ITERATIONS -> "tools"
      - otherwise -> "end" (final answer, question for the user, or iterations exhausted)
      - decide how the exhaustion case still produces a final answer (what was found / unresolved)
        e.g., a final forced LLM call without tools, or a dedicated "finalize" node
    """
    raise NotImplementedError("TODO (Member 5)")


def build_graph():
    """Create and compile the LangGraph StateGraph.

    TODO:
      - graph = StateGraph(AgentState)
      - add nodes "llm" (llm_node) and "tools" (tool_node)
      - set entry point to "llm"
      - add conditional edges from "llm" using should_continue: {"tools": "tools", "end": END}
      - add edge "tools" -> "llm"
      - return graph.compile()
    """
    raise NotImplementedError("TODO (Member 5)")


def run_agent(user_message: str) -> dict:
    """Convenience entry point used by the CLI: run the graph for one user message.

    TODO:
      - create the initial state, invoke the compiled graph
      - return {"answer": str, "iterations": int, "tool_results": list}
      - never raise to the caller (catch and return a friendly error answer)
    """
    raise NotImplementedError("TODO (Member 5)")