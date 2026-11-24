"""ChromaDB vector search tool.

Owner: Member 3 (feature/rag-retrieval).

Contract (see docs/contracts.md):
    success: {"status": "ok", "data": {"results": [{"text": str, "source": str, "score": float}]}}
    failure: {"status": "error", "error": "<clear message>"}
The function must NEVER raise to the agent. It must also work on its own (no agent needed).
"""
from __future__ import annotations


def search_documents(query: str, k: int = 3) -> dict:
    """Semantic search over the study-guide documents indexed in Chroma.

    Args:
        query: natural-language search text.
        k: number of results to return.

    Returns:
        {"status": "ok", "data": {"results": [{"text": str, "source": str, "score": float}]}}
        or {"status": "error", "error": "<clear message>"}

    TODO:
      - wrap the whole body in try/except and return the error dict on any failure
      - validate query (non-empty string) and k (positive int, sensible upper bound)
      - open the persistent Chroma collection at CHROMA_PATH (collection name from src/config.py)
      - embed the query with the SAME embedding model used in build_index.py
      - convert distances to a score (document how: higher = more relevant)
      - handle an empty/missing index with a clear error message
    """
    raise NotImplementedError("TODO (Member 3)")


if __name__ == "__main__":
    # TODO (Member 3): example call for running the tool independently:
    #   python -m src.tools.vector_tool
    print(search_documents("how to prepare for exams", k=3))