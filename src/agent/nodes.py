"""LangGraph nodes: llm_node and tool_node.

Owner: Member 5 (feature/agent-core).
"""
from __future__ import annotations

from src.agent.state import AgentState


def llm_node(state: AgentState) -> dict:
    """Call the LLM with the system prompt + messages + tool results.

    Each call is ONE iteration: the LLM analyzes the request and previous tool results, then
    picks an action: call a tool, ask the user for missing info, or write the final answer.

    TODO:
      - build the LLM with the tools bound (LLM_PROVIDER / LLM_API_KEY / LLM_MODEL from src/config.py)
      - prepend the system prompt from src/agent/prompts.py
      - increment state["iteration"]
      - log the iteration via src/agent/logger.py (iteration, short reasoning, action)
      - if iteration reached MAX_ITERATIONS, instruct the LLM to give a final answer stating what was
        found and what is unresolved (do not call more tools)
      - return {"messages": [ai_message], "iteration": new_iteration}
    """
    raise NotImplementedError("TODO (Member 5)")


def tool_node(state: AgentState) -> dict:
    """Execute the tool calls requested by the last AI message.

    TODO:
      - for each tool call, look up the tool (src/tools/__init__.py get_tool_map) and run it
      - tools never raise by contract, but still catch unexpected exceptions and convert them to
        {"status": "error", "error": "..."} so the agent never crashes
      - return results to the LLM as ToolMessage with JSON content (json.dumps of the result dict)
      - append {"iteration", "tool", "input", "output"} to state["tool_results"]
      - log tool input and output via src/agent/logger.py
      - return {"messages": [tool_messages], "tool_results": updated_list}
    """
    raise NotImplementedError("TODO (Member 5)")