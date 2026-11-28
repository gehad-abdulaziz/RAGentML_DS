"""Mock tools with the SAME signatures and return shapes as the real tools.

Deterministic fake data that follows docs/contracts.md. They need no model file,
no Neo4j and no Chroma, so the agent can be developed and tested anywhere.

Error injection:  set_force_error(True)  (or env MOCK_FORCE_ERROR=1) makes every
mock tool return {"status": "error", ...}.
"""
from __future__ import annotations

import os

_force_error: bool = os.getenv("MOCK_FORCE_ERROR", "0").strip().lower() in {"1", "true", "yes"}
FORCED_ERROR_MESSAGE = "Mock tool error injected for testing (MOCK_FORCE_ERROR)."

from src.tools.features import NUMERIC_RANGES as RANGES  # noqa: E402


def set_force_error(flag: bool) -> None:
    global _force_error
    _force_error = bool(flag)


def _error(message: str) -> dict:
    return {"status": "error", "error": message}


def mock_predict_performance(features: dict) -> dict:
    """Fake predict_performance: a simple formula so what-if runs give different numbers."""
    if _force_error:
        return _error(FORCED_ERROR_MESSAGE)
    if not isinstance(features, dict):
        return _error("features must be a dictionary.")
    missing = [k for k in ("Hours_Studied", "Attendance", "Previous_Scores", "Tutoring_Sessions")
               if features.get(k) is None]
    if missing:
        return _error(f"Missing required feature(s): {missing}")
    for name, (low, high) in RANGES.items():
        if name in features:
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return _error(f"Feature '{name}' must be numeric. Received: {value!r}.")
            if not low <= value <= high:
                return _error(f"Feature '{name}' must be between {low} and {high}.")
    score = (
        30
        + 0.6 * min(features["Hours_Studied"], 40)
        + 0.15 * features["Attendance"]
        + 0.2 * features["Previous_Scores"]
        + 0.5 * features["Tutoring_Sessions"]
    )
    score = round(max(0.0, min(100.0, score)), 2)
    return {"status": "ok",
            "data": {"prediction": score, "unit": "exam_score", "inputs_used": dict(features)}}


_CANNED_DOCS = [
    {"text": "Spaced repetition means reviewing material at increasing intervals.",
     "source": "mock_memory_learning.pdf", "title": "Memory and Learning", "page": 1,
     "chunk_id": "mock-1", "score": 0.91},
    {"text": "Active recall: close the book and test yourself instead of rereading.",
     "source": "mock_exam_preparation.pdf", "title": "Exam Preparation", "page": 2,
     "chunk_id": "mock-2", "score": 0.84},
    {"text": "Plan study blocks of 25-50 minutes with short breaks.",
     "source": "mock_time_management.pdf", "title": "Time Management", "page": 1,
     "chunk_id": "mock-3", "score": 0.77},
]


def mock_search_documents(query: str, k: int = 3) -> dict:
    """Fake search_documents: at most k canned results, error for an empty query."""
    if _force_error:
        return _error(FORCED_ERROR_MESSAGE)
    if not isinstance(query, str) or not query.strip():
        return _error("query must be a non-empty string.")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
        return _error("k must be an integer between 1 and 10.")
    return {"status": "ok", "data": {"results": [dict(d) for d in _CANNED_DOCS[:k]]}}


_CANNED_ROWS = {
    "prerequisites": [{"course": "Calculus II", "prerequisite": "Calculus I", "distance": 1}],
    "topics_in_course": [
        {"course": "Calculus I", "topic": "Limits", "description": "Mock topic"},
        {"course": "Calculus I", "topic": "Differentiation", "description": "Mock topic"},
    ],
    "resources_for_topic": [{"topic": "Limits", "resource": "Mock Limits Video",
                             "url": "https://example.com/limits"}],
    "courses_in_subject": [{"subject": "Mathematics", "course": "Calculus I"},
                           {"subject": "Mathematics", "course": "Calculus II"}],
    "strategies_for_topic": [{"topic": "Limits", "strategy": "Practice with graphs first"}],
    "topic_prerequisites": [{"topic": "Integration", "requires": "Differentiation"}],
}
_REQUIRED = {
    "prerequisites": "course", "topics_in_course": "course", "resources_for_topic": "topic",
    "courses_in_subject": "subject", "strategies_for_topic": "topic", "topic_prerequisites": "topic",
}


def mock_query_graph(query_type: str, params: dict) -> dict:
    """Fake query_graph: canned rows per query_type, errors for bad type / missing params."""
    if _force_error:
        return _error(FORCED_ERROR_MESSAGE)
    if query_type not in _CANNED_ROWS:
        return _error(f"Unknown query_type {query_type!r}. Valid query types: {sorted(_CANNED_ROWS)}")
    if not isinstance(params, dict):
        return _error("params must be a dict, e.g. {'course': 'Calculus II'}")
    needed = _REQUIRED[query_type]
    value = params.get(needed)
    if not isinstance(value, str) or not value.strip():
        return _error(f"Missing required param '{needed}' for query_type '{query_type}'")
    if value.strip().lower() == "does not exist":
        return {"status": "ok", "data": {"rows": []}}
    return {"status": "ok", "data": {"rows": [dict(r) for r in _CANNED_ROWS[query_type]]}}
