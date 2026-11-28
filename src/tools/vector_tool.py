"""ChromaDB vector search tool.

Owner: Member 3 (feature/rag-retrieval).

Contract (see docs/contracts.md):
    success: {"status": "ok", "data": {"results": [{"text": str, "source": str, "score": float, ...}]}}
    failure: {"status": "error", "error": "<clear message>"}
The function must NEVER raise to the agent. It must also work on its own (no agent needed).

How retrieval works
    1. The query is embedded with the SAME model that build_index.py used
       (EMBEDDING_MODEL_NAME), normalised the same way (normalize_embeddings=True).
    2. Chroma searches the "study_guides" collection, which was created with
       hnsw:space = cosine, so Chroma returns a cosine DISTANCE (0 = identical).
    3. score = 1 - distance = cosine similarity: in [-1, 1], higher = more relevant.
       Results are returned best first.
    4. source / title / page / chunk_id come from the chunk metadata written by
       build_index.py. Nothing is invented.

Run on its own:  python -m src.tools.vector_tool
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError:
    chromadb = None

from src import config

MAX_K = 10                # upper bound for k (keeps the LLM context small)
MAX_QUERY_CHARS = 1000    # longer text is rejected, not silently truncated
INDEX_HINT = "Build the index first with: python -m src.vectordb.build_index"

# Must match EMBEDDING_MODEL_NAME in src/vectordb/build_index.py and src/config.py
EMBEDDING_MODEL_NAME = getattr(config, "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

_embedding_model: Any = None   # loaded on first use, then reused
_clients: dict[str, Any] = {}  # one Chroma client per database path


def _error(message: str) -> dict:
    return {"status": "error", "error": message}


def _get_embedding_model() -> Any:
    """Load the sentence-transformers model once. The import is deferred on purpose."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _embed_query(query: str) -> list[float]:
    """Embed one query text exactly the way build_index.py embedded the chunks."""
    vector = _get_embedding_model().encode([query], normalize_embeddings=True)
    return vector[0].tolist()


def _get_client(path: Path) -> Any:
    key = str(path)
    if key not in _clients:
        _clients[key] = chromadb.PersistentClient(path=key)
    return _clients[key]


def _open_collection() -> tuple[Any, str | None]:
    """Return (collection, None) or (None, error message). Never creates a new index."""
    if chromadb is None:
        return None, "ChromaDB package is not installed. Install with: pip install chromadb"
    path = Path(config.CHROMA_PATH)
    # Check for the database file first. PersistentClient would silently create an
    # empty database in a missing folder, and that would hide the real problem.
    if not (path / "chroma.sqlite3").exists():
        return None, f"No ChromaDB index found at {path}. {INDEX_HINT}"
    try:
        collection = _get_client(path).get_collection(config.CHROMA_COLLECTION)
    except Exception as exc:
        return None, (
            f"Collection '{config.CHROMA_COLLECTION}' not found in {path} "
            f"({type(exc).__name__}). {INDEX_HINT}"
        )
    return collection, None


def search_documents(query: str, k: int = 3) -> dict:
    """Semantic search over the study-guide documents indexed in Chroma.

    Args:
        query: natural-language search text, in English or Arabic.
        k: number of results to return (1 to MAX_K).

    Returns:
        {"status": "ok", "data": {"results": [
            {"text": str, "source": str, "title": str, "page": int,
             "chunk_id": str, "score": float}, ...]}}
        or {"status": "error", "error": "<clear message>"}

        score: cosine similarity, higher = more relevant (1.0 = same direction).
        Fewer than k results are returned when the index holds fewer than k chunks.
    """
    try:
        # 1) validate the query
        if not isinstance(query, str):
            return _error(f"query must be a string, got {type(query).__name__}")
        text = query.strip()
        if not text:
            return _error("query is empty; pass a question or keywords to search for")
        if len(text) > MAX_QUERY_CHARS:
            return _error(
                f"query is too long ({len(text)} characters; the maximum is {MAX_QUERY_CHARS})"
            )

        # 2) validate k (bool is a subclass of int, so reject it explicitly)
        if isinstance(k, bool) or not isinstance(k, int):
            return _error(f"k must be a whole number from 1 to {MAX_K}, got {k!r}")
        if not 1 <= k <= MAX_K:
            return _error(f"k must be between 1 and {MAX_K}, got {k}")

        # 3) open the index
        collection, problem = _open_collection()
        if problem:
            return _error(problem)
        count = collection.count()
        if count == 0:
            return _error(
                f"The index in collection '{config.CHROMA_COLLECTION}' is empty. {INDEX_HINT}"
            )

        # 4) embed the query and search
        try:
            vector = _embed_query(text)
        except Exception as exc:
            return _error(
                f"Could not embed the query with the model '{EMBEDDING_MODEL_NAME}' "
                f"({type(exc).__name__}: {exc}). Check the internet connection on the "
                "first run, when the model is downloaded."
            )

        raw = collection.query(
            query_embeddings=[vector],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )

        # 5) format the results (best first)
        results = []
        for chunk_text, meta, distance in zip(
            raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            meta = meta or {}
            source = meta.get("source")
            if not source:
                continue  # every indexed chunk has a source; skip anything that does not
            results.append({
                "text": chunk_text,
                "source": source,
                "title": meta.get("title", ""),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id", ""),
                "score": round(1.0 - float(distance), 4),  # cosine distance -> similarity
            })

        return {"status": "ok", "data": {"results": results}}

    except Exception as exc:  # last safety net: the tool must NEVER raise to the agent
        return _error(f"Document search failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    # Run the tool on its own (no agent needed):  python -m src.tools.vector_tool
    for q in ["how to prepare for exams", "ما هو التكرار المتباعد؟"]:
        print(f"\nsearch_documents({q!r}, k=3)")
        print(json.dumps(search_documents(q, k=3), indent=2, ensure_ascii=False))
