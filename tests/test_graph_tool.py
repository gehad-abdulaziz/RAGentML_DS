"""Tests for the graph tool. Owner: Member 4 (feature/graph-tool).

Requires Neo4j running and seeded (docker compose up -d; python -m src.graphdb.seed_graph).
Run:  pytest tests/test_graph_tool.py -v
Save the output to logs/ (e.g., pytest ... | tee logs/graph_tool_tests.log).
"""
import json

import pytest

from src import config
from src.tools import graph_tool
from src.tools.graph_tool import QUERY_TYPES, query_graph


@pytest.fixture(autouse=True)
def fresh_driver():
    """Start and end every test with no cached driver (so config changes take effect)."""
    graph_tool.close_driver()
    yield
    graph_tool.close_driver()


@pytest.fixture
def neo4j_ready():
    """Skip DB tests with a clear reason if Neo4j is not running / not seeded."""
    result = query_graph("courses_in_subject", {"subject": "Mathematics"})
    if result["status"] != "ok" or not result["data"]["rows"]:
        pytest.skip("Neo4j is not running or not seeded (docker compose up -d; python -m src.graphdb.seed_graph)")


# Known seeded value for each query type -> a value we expect to find in the rows.
KNOWN_CASES = {
    "prerequisites": ({"course": "Calculus II"}, "prerequisite", "Calculus I"),
    "topics_in_course": ({"course": "Calculus I"}, "topic", "Limits"),
    "resources_for_topic": ({"topic": "Differentiation"}, "title", "Khan Academy - Calculus 1"),
    "courses_in_subject": ({"subject": "Mathematics"}, "course", "Calculus I"),
    "strategies_for_topic": ({"topic": "Limits"}, "strategy", "Active Recall"),
    "topic_prerequisites": ({"topic": "Integration"}, "requires", "Differentiation"),
}


def test_all_query_types_are_covered_by_tests():
    assert set(KNOWN_CASES) == set(QUERY_TYPES)


@pytest.mark.parametrize("query_type", sorted(KNOWN_CASES))
def test_each_query_type_returns_ok(neo4j_ready, query_type):
    params, column, expected = KNOWN_CASES[query_type]
    result = query_graph(query_type, params)
    assert result["status"] == "ok"
    assert isinstance(result["data"]["rows"], list)
    assert expected in [row[column] for row in result["data"]["rows"]]
    json.dumps(result)  # the result must be JSON-serializable


def test_indirect_prerequisites_have_distance(neo4j_ready):
    rows = query_graph("prerequisites", {"course": "Calculus II"})["data"]["rows"]
    distance = {r["prerequisite"]: r["distance"] for r in rows}
    assert distance["Calculus I"] == 1          # direct prerequisite
    assert distance["College Algebra"] == 2     # prerequisite of Calculus I


def test_lookup_is_case_insensitive(neo4j_ready):
    result = query_graph("topics_in_course", {"course": "  calculus i "})
    assert result["status"] == "ok"
    assert len(result["data"]["rows"]) > 0


def test_unknown_query_type_returns_error():
    result = query_graph("delete_everything", {})
    assert result["status"] == "error"
    assert "prerequisites" in result["error"]  # message lists the valid types


def test_missing_params_returns_error():
    result = query_graph("prerequisites", {})
    assert result["status"] == "error"
    assert "course" in result["error"]


def test_wrong_param_type_returns_error():
    assert query_graph("prerequisites", {"course": 123})["status"] == "error"
    assert query_graph("prerequisites", {"course": "   "})["status"] == "error"
    assert query_graph("prerequisites", "Calculus II")["status"] == "error"


def test_unknown_entity_returns_empty_rows(neo4j_ready):
    result = query_graph("prerequisites", {"course": "Underwater Basket Weaving"})
    assert result == {"status": "ok", "data": {"rows": []}}


def test_cypher_injection_attempt_is_harmless(neo4j_ready):
    attack = "x' }) DETACH DELETE c //"
    result = query_graph("prerequisites", {"course": attack})
    assert result == {"status": "ok", "data": {"rows": []}}  # treated as a plain (unknown) name
    # and the graph is still intact:
    after = query_graph("courses_in_subject", {"subject": "Mathematics"})
    assert len(after["data"]["rows"]) > 0


def test_neo4j_down_returns_error(monkeypatch):
    monkeypatch.setattr(config, "NEO4J_URI", "bolt://localhost:9")  # nothing listens here
    result = query_graph("prerequisites", {"course": "Calculus II"})  # must NOT raise
    assert result["status"] == "error"
    assert "Neo4j" in result["error"]
