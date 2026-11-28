"""LangGraph nodes for the academic advisor agent.

Owner: Member 5 (feature/agent-core).
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src import config
from src.agent.llm import get_llm
from src.agent.logger import log_iteration
from src.agent.prompts import MAX_ITERATIONS_PROMPT, SYSTEM_PROMPT
from src.agent.state import AgentState
from src.tools import get_tool_map, get_tools

# Initialize tools and registry
TOOLS = get_tools()
TOOLS_BY_NAME = get_tool_map()


def _get_agent_llm(bind_tools: bool = True):
    """Obtain the configured LLM and optionally bind available tools."""
    llm = get_llm()
    if bind_tools:
        # Re-fetch in case tools or mock mode changed
        tools = get_tools()
        return llm.bind_tools(tools)
    return llm


def llm_node(state: AgentState) -> dict[str, Any]:
    """Invoke the LLM bound with tools to decide next step.
    
    Increments iteration counter when tool calls are requested.
    """
    current_iter = state.get("iteration", 0)
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]

    llm = _get_agent_llm(bind_tools=True)
    response = llm.invoke(messages)

    has_tool_calls = bool(getattr(response, "tool_calls", None))

    if has_tool_calls:
        next_iter = current_iter + 1
        for call in response.tool_calls:
            log_iteration(
                iteration=next_iter,
                action="call_tool",
                reasoning="Agent requested tool execution.",
                tool_name=call.get("name"),
                tool_input=call.get("args"),
            )
        return {
            "messages": [response],
            "iteration": next_iter,
        }
    else:
        # No tool calls: either asking user for missing info or final answer
        content = str(getattr(response, "content", "")).strip()
        is_asking = "?" in content or any(
            phrase in content.lower()
            for phrase in ("please provide", "could you share", "how many hours", "what is your")
        )
        action = "ask_user" if is_asking else "final_answer"
        log_iteration(
            iteration=current_iter,
            action=action,
            reasoning=content[:120] if content else None,
        )
        return {
            "messages": [response],
            "iteration": current_iter,
        }


# Backwards compatibility alias
agent_node = llm_node


def force_final_node(state: AgentState) -> dict[str, Any]:
    """Invoked when maximum iterations (5) are reached.
    
    Requests a final plain-text answer without tool calls summarizing what was found
    and what remains unresolved, exactly per course requirements.
    """
    current_iter = state.get("iteration", config.MAX_ITERATIONS)
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]

    forced_prompt = (
        f"{MAX_ITERATIONS_PROMPT} "
        "Synthesize your final response for the student based on everything found so far. "
        "Explicitly summarize what was found and what remains unresolved. "
        "Write your final response in clear plain text without tool calls or JSON."
    )
    messages.append(HumanMessage(content=forced_prompt))
    llm = _get_agent_llm(bind_tools=False)
    response = llm.invoke(messages)

    log_iteration(
        iteration=current_iter,
        action="final_answer",
        reasoning="Forced final answer reached at iteration limit.",
    )
    return {"messages": [response], "iteration": current_iter}


def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute requested tools, record structured outputs, and return ToolMessages."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    current_iter = state.get("iteration", 1)
    new_tool_messages: list[ToolMessage] = []
    recorded_results: list[dict[str, Any]] = list(state.get("tool_results", []))

    tools_map = get_tool_map()

    for call in last_message.tool_calls:
        tool_name = call["name"]
        tool_args = call.get("args", {})
        tool_call_id = call["id"]

        tool = tools_map.get(tool_name)
        if tool is None:
            output = {"status": "error", "error": f"Unknown tool '{tool_name}'."}
        else:
            try:
                output = tool.invoke(tool_args)
            except Exception as exc:
                output = {"status": "error", "error": f"Tool execution failed: {str(exc)}"}

        recorded_results.append(
            {
                "iteration": current_iter,
                "tool": tool_name,
                "input": tool_args,
                "output": output,
            }
        )

        log_iteration(
            iteration=current_iter,
            action="call_tool",
            tool_name=tool_name,
            tool_input=tool_args,
            tool_output=output,
        )

        # Ensure content is a valid string representation of JSON
        if isinstance(output, str):
            content_str = output
        else:
            try:
                content_str = json.dumps(output, ensure_ascii=False)
            except Exception:
                content_str = str(output)

        new_tool_messages.append(
            ToolMessage(
                content=content_str,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    return {
        "messages": new_tool_messages,
        "tool_results": recorded_results,
    }
