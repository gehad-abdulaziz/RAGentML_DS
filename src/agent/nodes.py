"""LangGraph nodes for the academic advisor agent.

Graph:  START -> input_guard -> agent <-> tools -> (final_force) -> output_guard -> END

Iteration rule (documented decision): ONE iteration = one LLM reply that contains
tool calls (all calls of that reply run together). The 5th round still executes its
tools; afterwards the graph goes to `final_force`, which answers without tools.
So the LLM is called at most 6 times per turn and tools run at most 5 rounds.
"""
from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src import config
from src.agent import middleware as mw
from src.agent.llm import get_llm, uses_qwen3
from src.agent.logger import log_event, log_iteration
from src.agent.prompts import (
    EMPTY_REPLY_MESSAGE,
    LLM_DOWN_MESSAGE,
    MAX_ITERATIONS_PROMPT,
    NO_THINK_SUFFIX,
    SYSTEM_PROMPT,
)
from src.agent.state import AgentState
from src.tools import get_tool_map

MAX_TOOL_MESSAGE_CHARS = 6000


# ----------------------------------------------------------------- helpers

def _last_human_text(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _system_prompt(state: AgentState) -> str:
    prompt = SYSTEM_PROMPT + (state.get("memory_context") or "")
    if config.LLM_NO_THINK and uses_qwen3():
        prompt += NO_THINK_SUFFIX
    return prompt


def _with_content(message: AIMessage, content: str, tool_calls: list | None = None) -> AIMessage:
    update: dict[str, Any] = {"content": content}
    if tool_calls is not None:
        update["tool_calls"] = tool_calls
    try:
        return message.model_copy(update=update)
    except Exception:  # noqa: BLE001 - very old langchain-core
        return AIMessage(content=content, tool_calls=tool_calls if tool_calls is not None
                         else message.tool_calls, id=message.id)


def _ask_or_final(text: str) -> str:
    return "ask_user" if text.rstrip().endswith(("?", "؟")) else "final_answer"


def _ids(state: AgentState) -> dict[str, Any]:
    return {"user_id": state.get("user_id"), "session_id": state.get("session_id")}


def _invoke(llm: Any, messages: list, state: AgentState) -> AIMessage:
    """LLM call with retries; on total failure return a friendly message (no crash)."""
    def on_error(attempt: int, exc: Exception) -> None:
        log_event("llm_error", attempt=attempt, error=f"{type(exc).__name__}: {exc}"[:300], **_ids(state))

    try:
        return mw.call_llm_with_retry(llm, messages, on_error=on_error)
    except mw.LLMUnavailable:
        return AIMessage(content=LLM_DOWN_MESSAGE)


# ------------------------------------------------------------------- nodes

def input_guard_node(state: AgentState) -> dict[str, Any]:
    """Input guardrails: empty / too long / prompt injection / rate limit."""
    text = _last_human_text(state)
    result = mw.validate_user_input(text)
    if result.ok:
        allowed, retry_after = mw.rate_limiter.check(state.get("user_id", "anonymous"))
        if allowed:
            return {"blocked": False}
        reason = "rate_limited"
        message = (f"You are sending messages too fast. Please wait {int(retry_after) + 1}s. "
                   f"/ استنى {int(retry_after) + 1} ثانية وبعدين كمّل.")
    else:
        reason, message = result.reason, result.user_message

    log_event("guardrail_blocked", reason=reason, preview=mw.redact_sensitive(text)[:80], **_ids(state))
    log_iteration(state.get("iteration", 0), "blocked", reasoning=reason, **_ids(state))
    return {
        "blocked": True,
        "block_reason": reason,
        "messages": [AIMessage(content=message)],
        "guard_events": [*state.get("guard_events", []), f"input:{reason}"],
    }


def agent_node(state: AgentState) -> dict[str, Any]:
    """Ask the LLM what to do next: call tools, ask the student, or answer."""
    iteration = state.get("iteration", 0)
    messages = [SystemMessage(content=_system_prompt(state)), *state["messages"]]

    response = _invoke(get_llm(bind_tools=True), messages, state)
    content = mw.clean_model_output(response.content)
    tool_calls = list(getattr(response, "tool_calls", None) or [])

    if len(tool_calls) > config.MAX_TOOL_CALLS_PER_ITER:
        log_event("guardrail_tool_cap", requested=len(tool_calls),
                  allowed=config.MAX_TOOL_CALLS_PER_ITER, **_ids(state))
        tool_calls = tool_calls[: config.MAX_TOOL_CALLS_PER_ITER]

    if tool_calls:
        response = _with_content(response, content, tool_calls)
        return {"messages": [response], "iteration": iteration + 1}

    if not content:
        content = EMPTY_REPLY_MESSAGE
    response = _with_content(response, content, [])
    log_iteration(iteration, _ask_or_final(content), reasoning="LLM replied without tool calls",
                  tool_output={"answer_preview": content[:200]}, **_ids(state))
    return {"messages": [response], "iteration": iteration}


def tools_node(state: AgentState) -> dict[str, Any]:
    """Run the requested tools safely and record every call."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}

    iteration = state.get("iteration", 1)
    tool_map = get_tool_map()
    profile = state.get("profile_features") or {}
    reasoning = mw.clean_model_output(last.content) or None
    results: list[dict[str, Any]] = list(state.get("tool_results", []))
    tool_messages: list[ToolMessage] = []

    for call in last.tool_calls:
        name = call["name"]
        args = call.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        # small local models sometimes send features flat instead of {"features": {...}}
        if name == "predict_performance" and "features" not in args and args:
            args = {"features": dict(args)}

        effective = args
        from_profile: list[str] = []
        if name == "predict_performance" and isinstance(args.get("features", {}), dict):
            given = args.get("features", {})
            from_profile = [k for k in profile if k not in given]
            effective = {"features": {**profile, **given}}

        key = (name, json.dumps(effective, sort_keys=True, default=str))
        cached = next((r for r in results
                       if (r["tool"], json.dumps(r.get("effective_input", r["input"]),
                                                 sort_keys=True, default=str)) == key
                       and r["output"].get("status") == "ok"), None)

        started = time.monotonic()
        if cached is not None:
            output = cached["output"]
        elif name not in tool_map:
            output = {"status": "error",
                      "error": f"Unknown tool '{name}'. Valid tools: {sorted(tool_map)}."}
        else:
            output = mw.safe_tool_call(tool_map[name], effective)
            if output.get("status") == "ok" and from_profile and name == "predict_performance":
                output["data"]["from_profile"] = from_profile

        record = {
            "iteration": iteration, "tool": name, "input": args,
            "effective_input": effective, "output": output, "cached": cached is not None,
        }
        results.append(record)
        log_iteration(iteration, "call_tool", reasoning=reasoning, tool_name=name,
                      tool_input=args, tool_output=output, **_ids(state))
        log_event("tool_timing", tool=name, seconds=round(time.monotonic() - started, 3),
                  cached=cached is not None, **_ids(state))

        text = json.dumps(output, ensure_ascii=False, default=str)
        if len(text) > MAX_TOOL_MESSAGE_CHARS:
            text = text[:MAX_TOOL_MESSAGE_CHARS] + '..."truncated"}'
        tool_messages.append(ToolMessage(content=text, tool_call_id=call["id"], name=name))

    return {"messages": tool_messages, "tool_results": results}


def final_force_node(state: AgentState) -> dict[str, Any]:
    """Iteration limit reached: answer in plain text without tools."""
    messages = [
        SystemMessage(content=_system_prompt(state)),
        *state["messages"],
        HumanMessage(content=(f"{MAX_ITERATIONS_PROMPT} Write your final response in plain text "
                              "for the student. Do NOT output JSON or tool call syntax.")),
    ]
    response = _invoke(get_llm(bind_tools=False), messages, state)
    content = mw.clean_model_output(response.content) or EMPTY_REPLY_MESSAGE
    response = _with_content(response, content, [])
    log_iteration(state.get("iteration", config.MAX_ITERATIONS), "final_answer",
                  reasoning="MAX_ITERATIONS reached - forced final answer",
                  tool_output={"answer_preview": content[:200]}, **_ids(state))
    return {"messages": [response]}


def output_guard_node(state: AgentState) -> dict[str, Any]:
    """Output guardrails: redact secrets, flag ungrounded exam scores."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {}
    original = last.content if isinstance(last.content, str) else str(last.content)
    text = mw.redact_sensitive(mw.clean_model_output(original))

    warnings = mw.check_grounding(text, state.get("tool_results", []), _last_human_text(state))
    events = list(state.get("guard_events", []))
    if warnings:
        text += mw.GROUNDING_NOTE
        events.extend(f"output:{w}" for w in warnings)
        log_event("guardrail_ungrounded_score", warnings=warnings, **_ids(state))

    update: dict[str, Any] = {"guard_events": events}
    if text != original:
        update["messages"] = [AIMessage(content=text, id=last.id)]
    return update
