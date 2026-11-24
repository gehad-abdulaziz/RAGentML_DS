"""Tests for the vector search tool. Owner: Member 3 (feature/rag-retrieval).

Run:  pytest tests/test_vector_tool.py -v
Save the output to logs/ (e.g., pytest ... | tee logs/vector_tool_tests.log).
"""
import pytest  # noqa: F401

from src.tools.vector_tool import search_documents


def test_valid_query_returns_results():
    """TODO: status ok; results is a list of {"text", "source", "score"}; len(results) <= k."""
    pytest.skip("TODO (Member 3)")


def test_results_have_source_metadata():
    """TODO: every result has a non-empty source (needed so answers can cite real sources)."""
    pytest.skip("TODO (Member 3)")


def test_empty_query_returns_error():
    """TODO: empty / non-string query -> status error, never raises."""
    pytest.skip("TODO (Member 3)")


def test_invalid_k_returns_error():
    """TODO: k <= 0 or non-int -> status error."""
    pytest.skip("TODO (Member 3)")


def test_missing_index_returns_error():
    """TODO: point CHROMA_PATH to an empty dir (monkeypatch) -> clear error, no crash."""
    pytest.skip("TODO (Member 3)")