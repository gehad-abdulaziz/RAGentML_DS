"""LangGraph workflow definition.

Owner: Member 5 (feature/agent-core).
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src import config
from src.agent.nodes import agent_node, tools_node
from src.agent.state import AgentState, initial_state


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Route to 'tools' if the LLM issued tool calls; otherwise end."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def after_tools_route(state: AgentState) -> Literal["agent"]:
    """Always return to 'agent' after tool execution (agent handles MAX_ITERATIONS)."""
    return "agent"


def build_graph():
    """Compile and return the LangGraph academic advisor agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# Compiled graph instance ready to be imported by CLI / UI / tests
agent_graph = build_graph()


def run_agent(user_message: str) -> AgentState:
    """Helper to run a single user turn through the compiled graph."""
    state = initial_state(user_message)
    return agent_graph.invoke(state)


if __name__ == "__main__":
    # Quick standalone test with qwen3:4b
    test_query = "What are the prerequisites for Calculus II, and what topics are in Calculus I?"
    print(f"User: {test_query}\n")
    final_state = run_agent(test_query)
    print(f"Iterations used: {final_state['iteration']}")
    print(f"Tools called: {[r['tool'] for r in final_state['tool_results']]}")
    print(f"\nAssistant:\n{final_state['messages'][-1].content}")