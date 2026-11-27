"""Tests for the graph tool. Owner: Member 4 (feature/graph-tool).

Requires Neo4j running and seeded (docker compose up -d; python -m src.graphdb.seed_graph).
Run:  pytest tests/test_graph_tool.py -v
Save the output to logs/ (e.g., pytest ... | tee logs/graph_tool_tests.log).
"""
import pytest  # noqa: F401

from src.tools.graph_tool import query_graph


def test_each_query_type_returns_ok():
    """TODO: for every query type (prerequisites, topics_in_course, resources_for_topic,
    courses_in_subject) call with a known seeded value -> status ok, data["rows"] is a list."""
    pytest.skip("TODO (Member 4)")


def test_unknown_query_type_returns_error():
    """TODO: unknown query_type -> status error listing the valid types."""
    pytest.skip("TODO (Member 4)")


def test_missing_params_returns_error():
    """TODO: required param missing -> status error."""
    pytest.skip("TODO (Member 4)")


def test_unknown_entity_returns_empty_rows():
    """TODO: valid query, unknown course/topic -> status ok with rows == []."""
    pytest.skip("TODO (Member 4)")


def test_cypher_injection_attempt_is_harmless():
    """TODO: params containing quotes / Cypher fragments are treated as plain values."""
    pytest.skip("TODO (Member 4)")


def test_neo4j_down_returns_error():
    """TODO: bad NEO4J_URI (monkeypatch) -> status error, never raises."""
    pytest.skip("TODO (Member 4)")