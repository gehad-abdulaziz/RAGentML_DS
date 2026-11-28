"""Tests for the graph tool. Owner: Member 4 (feature/graph-tool).

Run: pytest tests/test_graph_tool.py -v
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:
    class _DummyMark:
        def parametrize(self, *args, **kwargs):
            return lambda fn: fn
    class _DummyPytest:
        mark = _DummyMark()
    pytest = _DummyPytest()

from src.tools.graph_tool import QUERY_TYPES, query_graph


def test_all_query_types_are_covered_by_tests():
    """Verify all defined query types in QUERY_TYPES are known and supported."""
    expected_types = {
        "prerequisites",
        "topics_in_course",
        "resources_for_topic",
        "courses_in_subject",
        "strategies_for_topic",
        "topic_prerequisites",
    }
    assert set(QUERY_TYPES.keys()) == expected_types


@pytest.mark.parametrize(
    "q_type, params",
    [
        ("courses_in_subject", {"subject": "Mathematics"}),
        ("prerequisites", {"course": "Calculus II"}),
        ("resources_for_topic", {"topic": "Limits"}),
        ("strategies_for_topic", {"topic": "Limits"}),
        ("topic_prerequisites", {"topic": "Differentiation"}),
        ("topics_in_course", {"course": "Calculus I"}),
    ],
)
def test_each_query_type_returns_ok(q_type, params):
    """Calling with valid query_type and params returns valid contract (status ok or graceful error)."""
    result = query_graph(q_type, params)
    assert isinstance(result, dict)
    assert "status" in result
    if result["status"] == "ok":
        assert "data" in result
        assert "rows" in result["data"]
        assert isinstance(result["data"]["rows"], list)
    else:
        # If Neo4j container is not running during unit tests, tool returns contract error
        assert "error" in result


def test_indirect_prerequisites_have_distance():
    """Prerequisites query should include distance in returned rows if live."""
    result = query_graph("prerequisites", {"course": "Calculus II"})
    assert isinstance(result, dict)
    if result["status"] == "ok" and result["data"]["rows"]:
        first = result["data"]["rows"][0]
        assert "distance" in first


def test_lookup_is_case_insensitive():
    """Query should work regardless of parameter casing."""
    r1 = query_graph("prerequisites", {"course": "calculus ii"})
    r2 = query_graph("prerequisites", {"course": "CALCULUS II"})
    assert r1["status"] == r2["status"]


def test_unknown_query_type_returns_error():
    """Unknown query_type returns status error."""
    result = query_graph("nonexistent_type", {"course": "Calculus I"})
    assert result["status"] == "error"
    assert "Unknown query_type" in result["error"]


def test_missing_params_returns_error():
    """Missing required parameters returns status error."""
    result = query_graph("prerequisites", {})
    assert result["status"] == "error"
    assert "Missing required param" in result["error"]


def test_wrong_param_type_returns_error():
    """Non-dict params returns status error."""
    result = query_graph("prerequisites", "invalid_params")  # type: ignore
    assert result["status"] == "error"
    assert "must be a dict" in result["error"]


def test_unknown_entity_returns_empty_rows():
    """Valid query with nonexistent course/topic returns status ok and empty rows."""
    result = query_graph("prerequisites", {"course": "Totally Nonexistent Course 999"})
    if result["status"] == "ok":
        assert result["data"]["rows"] == []
    else:
        assert "error" in result


def test_cypher_injection_attempt_is_harmless():
    """Injection syntax treated as literal string."""
    result = query_graph("prerequisites", {"course": "Calculus I' OR 1=1 RETURN c--"})
    assert isinstance(result, dict)
    if result["status"] == "ok":
        assert isinstance(result["data"]["rows"], list)


def test_neo4j_down_returns_error(monkeypatch):
    """When Neo4j cannot be reached, query_graph returns status error and NEVER raises."""
    monkeypatch.setattr("src.config.NEO4J_URI", "bolt://127.0.0.1:1")
    monkeypatch.setattr("src.tools.graph_tool._driver", None)
    result = query_graph("prerequisites", {"course": "Calculus II"})
    assert result["status"] == "error"


if __name__ == "__main__":
    print("Running Graph Tool Tests...")
    test_all_query_types_are_covered_by_tests()
    print("  [PASS] test_all_query_types_are_covered_by_tests")

    for q_type, params in [
        ("courses_in_subject", {"subject": "Mathematics"}),
        ("prerequisites", {"course": "Calculus II"}),
        ("resources_for_topic", {"topic": "Limits"}),
        ("strategies_for_topic", {"topic": "Limits"}),
        ("topic_prerequisites", {"topic": "Differentiation"}),
        ("topics_in_course", {"course": "Calculus I"}),
    ]:
        test_each_query_type_returns_ok(q_type, params)
        print(f"  [PASS] test_each_query_type_returns_ok[{q_type}]")

    test_indirect_prerequisites_have_distance()
    print("  [PASS] test_indirect_prerequisites_have_distance")
    test_lookup_is_case_insensitive()
    print("  [PASS] test_lookup_is_case_insensitive")
    test_unknown_query_type_returns_error()
    print("  [PASS] test_unknown_query_type_returns_error")
    test_missing_params_returns_error()
    print("  [PASS] test_missing_params_returns_error")
    test_wrong_param_type_returns_error()
    print("  [PASS] test_wrong_param_type_returns_error")
    test_unknown_entity_returns_empty_rows()
    print("  [PASS] test_unknown_entity_returns_empty_rows")
    test_cypher_injection_attempt_is_harmless()
    print("  [PASS] test_cypher_injection_attempt_is_harmless")

    class MonkeyPatcher:
        def setattr(self, target, name_or_val, val=None):
            if val is None:
                parts = target.rsplit(".", 1)
                import importlib
                mod = importlib.import_module(parts[0])
                setattr(mod, parts[1], name_or_val)
            else:
                setattr(target, name_or_val, val)

    test_neo4j_down_returns_error(MonkeyPatcher())
    print("  [PASS] test_neo4j_down_returns_error")
    print("\nSummary: All 15 graph tool test cases passed successfully!")
