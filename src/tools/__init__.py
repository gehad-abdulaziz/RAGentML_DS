"""Tool registry: exposes the three tools to the agent as LangChain StructuredTools.

Owner: Member 6 (feature/agent-interface).

Exactly three tools are exposed:
  1. predict_performance
  2. search_documents
  3. query_graph
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src import config
from src.agent.prompts import (
    GRAPH_TOOL_DESCRIPTION,
    ML_TOOL_DESCRIPTION,
    RAG_TOOL_DESCRIPTION,
)
from src.tools.mock_tools import (
    mock_predict_performance,
    mock_query_graph,
    mock_search_documents,
)

logger = logging.getLogger(__name__)


# Pydantic argument schemas for robust LangChain tool validation
class PredictPerformanceInput(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description=(
            "Dictionary containing student metrics. Key features include 'Hours_Studied' (0-100), "
            "'Attendance' (0-100), 'Sleep_Hours' (0-24), 'Previous_Scores' (0-100), "
            "'Tutoring_Sessions' (0-20), and 'Motivation_Level' ('Low'/'Medium'/'High')."
        ),
    )


class SearchDocumentsInput(BaseModel):
    query: str = Field(
        ...,
        description="Search text or question to find relevant study guides and learning strategies.",
    )
    k: int = Field(
        default=3,
        description="Number of relevant document chunks to retrieve (1 to 10).",
    )


class QueryGraphInput(BaseModel):
    query_type: str = Field(
        ...,
        description=(
            "Type of graph query. Must be one of: 'prerequisites', 'topics_in_course', "
            "'resources_for_topic', 'courses_in_subject', 'strategies_for_topic', 'topic_prerequisites'."
        ),
    )
    params: dict[str, Any] = Field(
        ...,
        description=(
            "Dictionary of parameters for the query. E.g. {'course': 'Calculus II'} for prerequisites, "
            "{'topic': 'Limits'} for resources_for_topic, {'subject': 'Mathematics'} for courses_in_subject."
        ),
    )


def _safe_wrapper(fn: Callable, tool_name: str) -> Callable:
    """Ensure tool execution NEVER raises an unhandled exception to the agent."""
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.error("Exception in tool '%s': %s", tool_name, exc, exc_info=True)
            return {
                "status": "error",
                "error": f"Tool '{tool_name}' failed with unexpected error: {str(exc)}",
            }
    wrapped.__name__ = getattr(fn, "__name__", tool_name)
    return wrapped


def _load_real_ml_tool() -> Callable:
    from src.tools.ml_tool import predict_performance
    return predict_performance


def _load_real_vector_tool() -> Callable:
    from src.tools.vector_tool import search_documents
    return search_documents


def _load_real_graph_tool() -> Callable:
    from src.tools.graph_tool import query_graph
    return query_graph


def get_tool_callables(use_mocks: bool | None = None) -> dict[str, Callable]:
    """Return raw callable functions for the three tools."""
    use_mock = config.USE_MOCK_TOOLS if use_mocks is None else use_mocks

    if use_mock:
        return {
            "predict_performance": mock_predict_performance,
            "search_documents": mock_search_documents,
            "query_graph": mock_query_graph,
        }

    # Attempt to load real tools; fall back to mocks gracefully if unavailable
    tools: dict[str, Callable] = {}
    try:
        tools["predict_performance"] = _load_real_ml_tool()
    except Exception:
        tools["predict_performance"] = mock_predict_performance

    try:
        tools["search_documents"] = _load_real_vector_tool()
    except Exception:
        tools["search_documents"] = mock_search_documents

    try:
        tools["query_graph"] = _load_real_graph_tool()
    except Exception:
        tools["query_graph"] = mock_query_graph

    return tools


def get_tools(use_mocks: bool | None = None) -> list[StructuredTool]:
    """Return the list of three LangChain StructuredTools."""
    callables = get_tool_callables(use_mocks=use_mocks)

    ml_tool = StructuredTool.from_function(
        func=_safe_wrapper(callables["predict_performance"], "predict_performance"),
        name="predict_performance",
        description=ML_TOOL_DESCRIPTION,
        args_schema=PredictPerformanceInput,
    )

    rag_tool = StructuredTool.from_function(
        func=_safe_wrapper(callables["search_documents"], "search_documents"),
        name="search_documents",
        description=RAG_TOOL_DESCRIPTION,
        args_schema=SearchDocumentsInput,
    )

    graph_tool = StructuredTool.from_function(
        func=_safe_wrapper(callables["query_graph"], "query_graph"),
        name="query_graph",
        description=GRAPH_TOOL_DESCRIPTION,
        args_schema=QueryGraphInput,
    )

    return [ml_tool, graph_tool, rag_tool]


def get_tool_map(use_mocks: bool | None = None) -> dict[str, StructuredTool]:
    """Return {tool_name: StructuredTool} for dispatch."""
    tools = get_tools(use_mocks=use_mocks)
    return {tool.name: tool for tool in tools}
