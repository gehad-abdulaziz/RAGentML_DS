"""Mock tools with the SAME signatures and return shapes as the real tools.

Owner: Member 6 (feature/agent-interface). Used by Members 5 and 6 until the real
tools are merged into develop.

Follow docs/contracts.md exactly. Mocks should be deterministic and return realistic
(but clearly fake) data. Include a way to simulate errors so the error path can be tested.
"""
from __future__ import annotations


def mock_predict_performance(features: dict) -> dict:
    """Fake version of ml_tool.predict_performance.

    TODO:
      - return {"status": "ok", "data": {"prediction": <float derived from features>,
                "unit": "exam score (0-100)", "inputs_used": features}}
      - return {"status": "error", "error": "..."} when features is invalid or a required
        feature is missing (mimic the real validation)
      - make the prediction depend on e.g. Hours_Studied so what-if scenarios look different
    """
    raise NotImplementedError("TODO (Member 6)")


def mock_search_documents(query: str, k: int = 3) -> dict:
    """Fake version of vector_tool.search_documents.

    TODO:
      - return {"status": "ok", "data": {"results": [{"text": str, "source": str, "score": float}]}}
        with at most k canned results
      - return an error dict for an empty query
    """
    raise NotImplementedError("TODO (Member 6)")


def mock_query_graph(query_type: str, params: dict) -> dict:
    """Fake version of graph_tool.query_graph.

    TODO:
      - return {"status": "ok", "data": {"rows": [...]}} with canned rows per query_type
      - return an error dict for an unknown query_type or missing params
    """
    raise NotImplementedError("TODO (Member 6)")


# TODO (Member 6): optional switch to force errors, e.g. MOCK_FORCE_ERROR = False,
# for error-injection tests.