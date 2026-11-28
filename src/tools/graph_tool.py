"""Neo4j graph tool.

Owner: Member 4 (feature/graph-tool).

Contract (see docs/contracts.md):
    success: {"status": "ok", "data": {"rows": [...]}}
    failure: {"status": "error", "error": "<clear message>"}
The function NEVER raises to the agent. It also works on its own (no agent needed):
    python -m src.tools.graph_tool
"""
from __future__ import annotations

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable
except ImportError:
    GraphDatabase = None

    class AuthError(Exception):
        pass

    class ServiceUnavailable(Exception):
        pass

from src import config

# Fixed set of supported queries. Each maps to ONE parameterized Cypher statement.
# - `$course` / `$topic` / `$subject` are PARAMETERS: Neo4j treats them as plain values,
#   so user text can never change the meaning of the query (no Cypher injection).
# - toLower(...) = toLower($x) makes the match case-insensitive ("calculus i" finds "Calculus I").
# - Arrow direction:  A-[:PREREQUISITE_OF]->B  means A is taken BEFORE B.
#                     A-[:REQUIRES]->B         means B must be learned BEFORE A.
QUERY_TYPES: dict[str, dict] = {
    # All courses that must be taken before the given course (direct and indirect).
    # distance 1 = direct prerequisite, 2 = prerequisite of a prerequisite, ...
    "prerequisites": {
        "required_params": ["course"],
        "cypher": (
            "MATCH path = (p:Course)-[:PREREQUISITE_OF*1..5]->(c:Course) "
            "WHERE toLower(c.name) = toLower($course) "
            "RETURN c.name AS course, p.name AS prerequisite, min(length(path)) AS distance "
            "ORDER BY distance, prerequisite"
        ),
    },
    # Topics taught in a course.
    "topics_in_course": {
        "required_params": ["course"],
        "cypher": (
            "MATCH (c:Course)-[:HAS_TOPIC]->(t:Topic) "
            "WHERE toLower(c.name) = toLower($course) "
            "RETURN c.name AS course, t.name AS topic, t.description AS description "
            "ORDER BY topic"
        ),
    },
    # Learning resources (with real links) that cover a topic.
    "resources_for_topic": {
        "required_params": ["topic"],
        "cypher": (
            "MATCH (r:Resource)-[:COVERS]->(t:Topic) "
            "WHERE toLower(t.name) = toLower($topic) "
            "RETURN t.name AS topic, r.title AS title, r.url AS url, r.type AS type "
            "ORDER BY title"
        ),
    },
    # Courses that belong to a subject.
    "courses_in_subject": {
        "required_params": ["subject"],
        "cypher": (
            "MATCH (c:Course)-[:BELONGS_TO]->(s:Subject) "
            "WHERE toLower(s.name) = toLower($subject) "
            "RETURN s.name AS subject, c.name AS course, c.code AS code "
            "ORDER BY course"
        ),
    },
    # (extra) Study strategies that work best for a topic.
    "strategies_for_topic": {
        "required_params": ["topic"],
        "cypher": (
            "MATCH (t:Topic)-[:BEST_STUDIED_WITH]->(st:Strategy) "
            "WHERE toLower(t.name) = toLower($topic) "
            "RETURN t.name AS topic, st.name AS strategy, st.description AS description "
            "ORDER BY strategy"
        ),
    },
    # (extra) Topics that must be learned before the given topic (direct and indirect).
    "topic_prerequisites": {
        "required_params": ["topic"],
        "cypher": (
            "MATCH path = (t:Topic)-[:REQUIRES*1..5]->(req:Topic) "
            "WHERE toLower(t.name) = toLower($topic) "
            "RETURN t.name AS topic, req.name AS requires, min(length(path)) AS distance "
            "ORDER BY distance, requires"
        ),
    },
}

# One driver is created on first use and reused (creating a driver per call is slow).
_driver = None


def _get_driver():
    """Return the shared driver, creating it on the first call."""
    global _driver
    if GraphDatabase is None:
        return None
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            connection_timeout=5,  # seconds; fail fast when Neo4j is down
        )
    return _driver


def close_driver() -> None:
    """Close and forget the shared driver (used by tests, e.g. after changing NEO4J_URI)."""
    global _driver
    if _driver is not None:
        try:
            _driver.close()
        finally:
            _driver = None


def _error(message: str) -> dict:
    return {"status": "error", "error": message}


def query_graph(query_type: str, params: dict) -> dict:
    """Run one of the predefined graph queries.

    Args:
        query_type: one of QUERY_TYPES keys, e.g. "prerequisites".
        params: query parameters, e.g. {"course": "Calculus II"}.

    Returns:
        {"status": "ok", "data": {"rows": [...]}}
        or {"status": "error", "error": "<clear message>"}
    """
    try:
        valid_types = sorted(QUERY_TYPES)

        # 1) validate query_type
        if not isinstance(query_type, str) or query_type not in QUERY_TYPES:
            return _error(f"Unknown query_type {query_type!r}. Valid query types: {valid_types}")

        # 2) validate params
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _error("params must be a dict, e.g. {'course': 'Calculus II'}")
        spec = QUERY_TYPES[query_type]
        clean_params = {}
        for name in spec["required_params"]:
            value = params.get(name)
            if value is None:
                return _error(f"Missing required param '{name}' for query_type '{query_type}'")
            if not isinstance(value, str) or not value.strip():
                return _error(f"Param '{name}' must be a non-empty string")
            clean_params[name] = value.strip()

        # 3) run the parameterized Cypher; read the rows INSIDE the session
        driver = _get_driver()
        if driver is None:
            return _error(
                "Graph database driver is not installed. Install with: pip install neo4j"
            )
        with driver.session() as session:
            result = session.run(spec["cypher"], **clean_params)
            rows = [record.data() for record in result]

        # no match -> ok with empty rows (an unknown entity is not an error)
        return {"status": "ok", "data": {"rows": rows}}

    except (ServiceUnavailable, AuthError, OSError) as exc:
        # Neo4j is down, unreachable, or the password in .env is wrong
        return _error(
            "Graph database is unreachable or the credentials are wrong "
            f"({type(exc).__name__}). Check that Neo4j is running (docker compose up -d) "
            "and that NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env are correct."
        )
    except Exception as exc:  # last safety net: the tool must NEVER raise to the agent
        return _error(f"Graph query failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    # Run the tool on its own (no agent needed):  python -m src.tools.graph_tool
    import json

    for qt, p in [
        ("prerequisites", {"course": "Calculus II"}),
        ("topics_in_course", {"course": "Calculus I"}),
        ("resources_for_topic", {"topic": "Differentiation"}),
        ("courses_in_subject", {"subject": "Mathematics"}),
        ("strategies_for_topic", {"topic": "Limits"}),
        ("topic_prerequisites", {"topic": "Integration"}),
        ("prerequisites", {"course": "Does Not Exist"}),  # ok with empty rows
        ("bad_type", {}),  # error
    ]:
        print(f"\n{qt} {p}")
        print(json.dumps(query_graph(qt, p), indent=2, ensure_ascii=False))
    close_driver()
