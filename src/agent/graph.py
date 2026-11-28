"""LangGraph workflow definition.

Owner: Member 5 (feature/agent-core).
"""
from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src import config
from src.agent.nodes import force_final_node, llm_node, tool_node
from src.agent.state import AgentState, initial_state


def should_continue(state: AgentState) -> Literal["tools", "force_final", "__end__"]:
    """Conditional edge from llm_node.
    
    Per PDF page 6 specifications:
    - Tool requested AND iteration < 5  -> 'tools'
    - Tool requested AND iteration >= 5 -> 'force_final'
    - No tool requested (ask user or final answer) -> END
    """
    last_message = state["messages"][-1]
    has_tools = bool(getattr(last_message, "tool_calls", None))
    current_iter = state.get("iteration", 0)

    if has_tools:
        if current_iter >= config.MAX_ITERATIONS:
            return "force_final"
        return "tools"
    return END


def build_graph():
    """Compile and return the LangGraph academic advisor agent.
    
    Matches PDF page 6 workflow:
      llm -> (conditional) -> tools -> llm
      طلب أداة و iteration < 5  <- tools
      طلب أداة و iteration >= 5 <- force_final
      من غير tool call          <- END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("llm", llm_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("force_final", force_final_node)

    workflow.add_edge(START, "llm")
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "force_final": "force_final",
            END: END,
        },
    )
    workflow.add_edge("tools", "llm")
    workflow.add_edge("force_final", END)

    return workflow.compile()


# Compiled graph instance
agent_graph = build_graph()


def run_agent(
    input_data: str | list[BaseMessage] | AgentState,
    previous_messages: list[BaseMessage] | None = None,
) -> AgentState:
    """Run a user query through the compiled agent graph with conversation history support.

    Args:
        input_data: Either a user query string, a list of BaseMessages, or an AgentState dict.
        previous_messages: Optional list of previous conversation messages for multi-turn chat.

    Returns:
        The final AgentState containing the message history and tool execution results.
    """
    if isinstance(input_data, str):
        if previous_messages:
            messages = list(previous_messages) + [HumanMessage(content=input_data)]
        else:
            messages = [HumanMessage(content=input_data)]
        state: AgentState = {
            "messages": messages,
            "tool_results": [],
            "iteration": 0,
        }
    elif isinstance(input_data, list):
        state = {
            "messages": list(input_data),
            "tool_results": [],
            "iteration": 0,
        }
    elif isinstance(input_data, dict) and "messages" in input_data:
        state = input_data  # type: ignore[assignment]
    else:
        raise ValueError(f"Invalid input_data type: {type(input_data)}")

    return agent_graph.invoke(state)


if __name__ == "__main__":
    test_query = "What are the prerequisites for Calculus II, and what topics are in Calculus I?"
    print(f"User: {test_query}\n")
    final_state = run_agent(test_query)
    print(f"Iterations used: {final_state['iteration']}")
    print(f"Tools called: {[r['tool'] for r in final_state['tool_results']]}")
    print(f"\nAssistant:\n{final_state['messages'][-1].content}")
