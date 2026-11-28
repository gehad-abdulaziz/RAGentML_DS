"""Tool registry: exposes exactly three tools to the agent as LangChain tools.

    predict_performance   (ML)      search_documents (RAG)      query_graph (Neo4j)

* Real implementations are imported lazily, so mock mode works without joblib,
  chromadb, sentence-transformers or neo4j being usable.
* Switch mock <-> real with USE_MOCK_TOOLS=1 in .env, `--mock` in the CLI, or
  set_mock_mode(True) in tests. The choice is read on every call.

NOTE: the spec calls this file "init.py"; in Python it must be __init__.py.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src import config
from src.agent.prompts import GRAPH_TOOL_DESCRIPTION, ML_TOOL_DESCRIPTION, RAG_TOOL_DESCRIPTION
from src.tools.features import ALL_FEATURES, DEFAULT_SECONDARY, KEY_FEATURES, normalize_features

_mock_mode: bool | None = None  # None -> follow config.USE_MOCK_TOOLS


def set_mock_mode(flag: bool | None) -> None:
    """Force mock (True) / real (False) tools; None goes back to the .env setting."""
    global _mock_mode
    _mock_mode = flag


def is_mock_mode() -> bool:
    return config.USE_MOCK_TOOLS if _mock_mode is None else _mock_mode


# ---------------------------------------------------------------- argument schemas

class PredictArgs(BaseModel):
    features: dict[str, Any] = Field(
        description="Student attributes, e.g. {'Hours_Studied': 20, 'Attendance': 90, "
                    "'Previous_Scores': 80, 'Tutoring_Sessions': 1}"
    )


class SearchArgs(BaseModel):
    query: str = Field(description="Natural-language question or keywords (English or Arabic).")
    k: int = Field(default=3, ge=1, le=10, description="Number of chunks to return (1-10).")


class GraphArgs(BaseModel):
    query_type: str = Field(description="One of: prerequisites, topics_in_course, "
                                        "resources_for_topic, courses_in_subject, "
                                        "strategies_for_topic, topic_prerequisites")
    params: dict[str, Any] = Field(default_factory=dict,
                                   description="e.g. {'course': 'Calculus II'} or {'topic': 'Limits'}")


# ----------------------------------------------------------------- dispatch layer

def _ml(features: dict) -> dict:
    if is_mock_mode():
        from src.tools.mock_tools import mock_predict_performance
        return mock_predict_performance(features)
    from src.tools.ml_tool import predict_performance
    return predict_performance(features)


def _search(query: str, k: int = 3) -> dict:
    if is_mock_mode():
        from src.tools.mock_tools import mock_search_documents
        return mock_search_documents(query, k)
    from src.tools.vector_tool import search_documents
    return search_documents(query, k)


def _graph(query_type: str, params: dict | None = None) -> dict:
    if is_mock_mode():
        from src.tools.mock_tools import mock_query_graph
        return mock_query_graph(query_type, params or {})
    from src.tools.graph_tool import query_graph
    return query_graph(query_type, params or {})


def predict_student_score(features: dict) -> dict:
    """Agent-facing wrapper around the ML tool.

    * The 4 KEY_FEATURES are required (no silent guessing).
    * Optional features fall back to DEFAULT_SECONDARY; the used defaults are returned
      as data["assumed_defaults"] so the answer can disclose them.
    * Unknown feature names are reported in data["ignored_inputs"].
    """
    if not isinstance(features, dict):
        return {"status": "error", "error": "features must be a dictionary."}

    features = normalize_features(features)
    missing = [k for k in KEY_FEATURES if features.get(k) in (None, "")]
    if missing:
        return {
            "status": "error",
            "error": (
                f"Missing required feature(s): {missing}. Ask the student for these exact "
                "values. Do not guess them."
            ),
        }

    known = {k: v for k, v in features.items() if k in ALL_FEATURES}
    ignored = sorted(k for k in features if k not in ALL_FEATURES)
    assumed = {k: v for k, v in DEFAULT_SECONDARY.items() if k not in known}
    result = _ml({**DEFAULT_SECONDARY, **known})
    if result.get("status") == "ok":
        data = result["data"]
        data["assumed_defaults"] = assumed
        if ignored:
            data["ignored_inputs"] = ignored
    return result


# ---------------------------------------------------------------------- registry

_TOOLS: list[StructuredTool] | None = None


def get_tools() -> list[StructuredTool]:
    """Return the three LangChain tools (exactly three) for the agent."""
    global _TOOLS
    if _TOOLS is None:
        _TOOLS = [
            StructuredTool.from_function(
                func=predict_student_score, name="predict_performance",
                description=ML_TOOL_DESCRIPTION, args_schema=PredictArgs),
            StructuredTool.from_function(
                func=_graph, name="query_graph",
                description=GRAPH_TOOL_DESCRIPTION, args_schema=GraphArgs),
            StructuredTool.from_function(
                func=_search, name="search_documents",
                description=RAG_TOOL_DESCRIPTION, args_schema=SearchArgs),
        ]
    return _TOOLS


def get_tool_map() -> dict[str, StructuredTool]:
    """Return {tool_name: tool} for the tool node to dispatch calls."""
    return {tool.name: tool for tool in get_tools()}
