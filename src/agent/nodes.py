"""LangGraph nodes for the academic advisor agent.

Owner: Member 5 (feature/agent-core).
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

from src import config
from src.agent.prompts import (
    GRAPH_TOOL_DESCRIPTION,
    MAX_ITERATIONS_PROMPT,
    ML_TOOL_DESCRIPTION,
    RAG_TOOL_DESCRIPTION,
    SYSTEM_PROMPT,
)
from src.agent.state import AgentState
from src.tools.graph_tool import query_graph
from src.tools.ml_tool import predict_performance
from src.tools.vector_tool import search_documents

# Baseline defaults for secondary features so students only provide key metrics
DEFAULT_STUDENT_FEATURES: dict[str, Any] = {
    "Hours_Studied": 15,
    "Attendance": 85,
    "Parental_Involvement": "Medium",
    "Access_to_Resources": "Medium",
    "Extracurricular_Activities": "Yes",
    "Sleep_Hours": 7,
    "Previous_Scores": 75,
    "Motivation_Level": "Medium",
    "Internet_Access": "Yes",
    "Tutoring_Sessions": 1,
    "Family_Income": "Medium",
    "Teacher_Quality": "Medium",
    "School_Type": "Public",
    "Peer_Influence": "Neutral",
    "Physical_Activity": 3,
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "College",
    "Distance_from_Home": "Near",
    "Gender": "Male",
}


def predict_student_score(features: dict[str, Any]) -> dict[str, Any]:
    """Wrapper around Member 1's ML tool that fills unmentioned features with baseline defaults."""
    if not isinstance(features, dict):
        return {"status": "error", "error": "features must be a dictionary."}

    merged_features = {**DEFAULT_STUDENT_FEATURES, **features}
    defaulted_keys = [k for k in DEFAULT_STUDENT_FEATURES if k not in features]

    result = predict_performance(merged_features)
    if result.get("status") == "ok":
        result["data"]["assumed_defaults"] = defaulted_keys
    return result


# Wrap team functions as LangChain StructuredTools with rich descriptions
ml_langchain_tool = StructuredTool.from_function(
    func=predict_student_score,
    name="predict_performance",
    description=ML_TOOL_DESCRIPTION,
)

graph_langchain_tool = StructuredTool.from_function(
    func=query_graph,
    name="query_graph",
    description=GRAPH_TOOL_DESCRIPTION,
)

rag_langchain_tool = StructuredTool.from_function(
    func=search_documents,
    name="search_documents",
    description=RAG_TOOL_DESCRIPTION,
)

TOOLS = [ml_langchain_tool, graph_langchain_tool, rag_langchain_tool]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def _get_llm(bind_tools: bool = True):
    """Initialize the local Ollama model (qwen3:4b by default)."""
    model_name = config.LLM_MODEL or "qwen3:4b"
    llm = ChatOllama(
        model=model_name,
        temperature=0.1,  # low temperature for reliable JSON tool arguments
    )
    return llm.bind_tools(TOOLS) if bind_tools else llm


def _needs_ml_guardrail(state: AgentState, response: AIMessage) -> bool:
    """Detect if the model tried to guess an exam score without calling predict_performance."""
    if getattr(response, "tool_calls", None):
        return False

    tools_used = {r["tool"] for r in state.get("tool_results", [])}
    if "predict_performance" in tools_used:
        return False

    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    ).lower()

    ml_keywords = ("predict", "exam score", "what score", "predicted score")
    return any(kw in last_user_msg for kw in ml_keywords)


def agent_node(state: AgentState) -> dict[str, Any]:
    """Invoke the LLM to decide the next step (tool call or final answer)."""
    current_iter = state.get("iteration", 0)
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]

    # If max iterations reached, force a final plain-text answer without tools
    if current_iter >= config.MAX_ITERATIONS:
        messages.append(
            HumanMessage(
                content=(
                    f"{MAX_ITERATIONS_PROMPT} "
                    "Write your final response in plain text for the student. "
                    "Do NOT output JSON or tool call syntax."
                )
            )
        )
        llm = _get_llm(bind_tools=False)
        response = llm.invoke(messages)
        return {"messages": [response]}

    llm = _get_llm(bind_tools=True)
    response = llm.invoke(messages)

    # Guardrail: prevent hallucinated scores when predict_performance was skipped
    if _needs_ml_guardrail(state, response):
        correction_messages = [
            *messages,
            response,
            HumanMessage(
                content=(
                    "SYSTEM CHECK: You have NOT called `predict_performance` yet. "
                    "Do NOT guess or calculate the score yourself. Call `predict_performance` "
                    "now with the student's mentioned features before giving the final answer."
                )
            ),
        ]
        response = llm.invoke(correction_messages)

    # Increment iteration only when the model decides to execute tools
    next_iter = current_iter + 1 if getattr(response, "tool_calls", None) else current_iter
    return {
        "messages": [response],
        "iteration": next_iter,
    }


def tools_node(state: AgentState) -> dict[str, Any]:
    """Execute requested tools, record structured outputs, and return ToolMessages."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    current_iter = state.get("iteration", 1)
    new_tool_messages: list[ToolMessage] = []
    recorded_results: list[dict[str, Any]] = list(state.get("tool_results", []))

    for call in last_message.tool_calls:
        tool_name = call["name"]
        tool_args = call.get("args", {})
        tool_call_id = call["id"]

        tool = TOOLS_BY_NAME.get(tool_name)
        if tool is None:
            output = {"status": "error", "error": f"Unknown tool '{tool_name}'."}
        else:
            output = tool.invoke(tool_args)

        recorded_results.append(
            {
                "iteration": current_iter,
                "tool": tool_name,
                "input": tool_args,
                "output": output,
            }
        )

        new_tool_messages.append(
            ToolMessage(
                content=json.dumps(output, ensure_ascii=False),
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    return {
        "messages": new_tool_messages,
        "tool_results": recorded_results,
    }