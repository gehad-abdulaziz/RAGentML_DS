"""LangGraph workflow definition.

    START -> input_guard --blocked--> END
                 |
                 v
              agent --tool calls--> tools --iteration < 5--> agent
                 |                    |
                 | no tool calls      +--iteration >= 5--> final_force
                 v                                             |
            output_guard <-------------------------------------+
                 |
                 v
                END
"""
from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

from src import config
from src.agent.nodes import (
    agent_node,
    final_force_node,
    input_guard_node,
    output_guard_node,
    tools_node,
)
from src.agent.state import AgentState, initial_state


def route_after_input(state: AgentState) -> Literal["agent", "end"]:
    return "end" if state.get("blocked") else "agent"


def route_after_agent(state: AgentState) -> Literal["tools", "output_guard"]:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "output_guard"


def route_after_tools(state: AgentState) -> Literal["agent", "final_force"]:
    return "final_force" if state.get("iteration", 0) >= config.MAX_ITERATIONS else "agent"


def build_graph():
    """Compile and return the LangGraph academic advisor agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("final_force", final_force_node)
    workflow.add_node("output_guard", output_guard_node)

    workflow.add_edge(START, "input_guard")
    workflow.add_conditional_edges("input_guard", route_after_input,
                                   {"agent": "agent", "end": END})
    workflow.add_conditional_edges("agent", route_after_agent,
                                   {"tools": "tools", "output_guard": "output_guard"})
    workflow.add_conditional_edges("tools", route_after_tools,
                                   {"agent": "agent", "final_force": "final_force"})
    workflow.add_edge("final_force", "output_guard")
    workflow.add_edge("output_guard", END)

    return workflow.compile()


# Compiled graph instance ready to be imported by CLI / UI / tests / LangGraph Studio
agent_graph = build_graph()


def run_agent(
    user_message: str,
    history: list[BaseMessage] | None = None,
    user_id: Any = "anonymous",
    session_id: str = "default",
    user_name: str = "",
    memory_context: str = "",
    profile_features: dict[str, Any] | None = None,
) -> AgentState:
    """Run ONE user turn. Pass `history` (previous messages) to continue a conversation."""
    from src.agent.middleware import clean_text

    state = initial_state(clean_text(user_message), history, user_id, session_id,
                          user_name, memory_context, profile_features)
    return agent_graph.invoke(state, config={"recursion_limit": 40})


if __name__ == "__main__":
    query = "What are the prerequisites for Calculus II, and what topics are in Calculus I?"
    print(f"User: {query}\n")
    final = run_agent(query)
    print(f"Iterations used: {final['iteration']}")
    print(f"Tools called: {[r['tool'] for r in final['tool_results']]}")
    print(f"\nAssistant:\n{final['messages'][-1].content}")
