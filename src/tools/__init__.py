"""Tool registry: exposes the three tools to the agent as LangChain tools.

Owner: Member 6 (feature/agent-interface).

NOTE: the spec calls this file "init.py"; in Python it must be named __init__.py to be
a package initializer. Rename it if your team prefers a different registry module.

Exactly three tools are exposed: predict_performance, search_documents, query_graph.
"""
from __future__ import annotations

# TODO (Member 6): switch between mock and real tools with one flag, e.g.
#   USE_MOCK_TOOLS = os.getenv("USE_MOCK_TOOLS", "0") == "1"
# Mocks live in src/tools/mock_tools.py; real tools in ml_tool.py, vector_tool.py, graph_tool.py.


def get_tools() -> list:
    """Return the list of LangChain tools (exactly three) for the agent.

    TODO:
      - wrap each function with langchain_core.tools.tool (or StructuredTool)
      - take the descriptions from src/agent/prompts.py so the LLM knows when to use each tool
      - tool names: "predict_performance", "search_documents", "query_graph"
      - the wrapped tools must return the contract dict (status/data|error) as JSON-serializable
    """
    raise NotImplementedError("TODO (Member 6)")


def get_tool_map() -> dict:
    """Return {tool_name: callable} for the tool node to dispatch calls.

    TODO: build from get_tools().
    """
    raise NotImplementedError("TODO (Member 6)")