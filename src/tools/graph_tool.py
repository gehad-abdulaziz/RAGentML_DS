"""Neo4j graph tool.

Owner: Member 4 (feature/graph-tool).

Contract (see docs/contracts.md):
    success: {"status": "ok", "data": {"rows": [...]}}
    failure: {"status": "error", "error": "<clear message>"}
The function must NEVER raise to the agent. It must also work on its own (no agent needed).
"""
from __future__ import annotations

# Fixed set of supported queries. Each maps to ONE parameterized Cypher statement.
# TODO (Member 4): fill in the Cypher and the required params for each query type.
QUERY_TYPES: dict[str, dict] = {
    "prerequisites": {"required_params": ["course"], "cypher": ""},  # TODO
    "topics_in_course": {"required_params": ["course"], "cypher": ""},  # TODO
    "resources_for_topic": {"required_params": ["topic"], "cypher": ""},  # TODO
    "courses_in_subject": {"required_params": ["subject"], "cypher": ""},  # TODO
    # TODO (optional): "strategies_for_topic" using (Topic)-[:BEST_STUDIED_WITH]->(Strategy)
}


def query_graph(query_type: str, params: dict) -> dict:
    """Run one of the predefined graph queries.

    Args:
        query_type: one of QUERY_TYPES keys, e.g. "prerequisites".
        params: query parameters, e.g. {"course": "Machine Learning"}.

    Returns:
        {"status": "ok", "data": {"rows": [...]}}
        or {"status": "error", "error": "<clear message>"}

    TODO:
      - wrap the whole body in try/except and return the error dict on any failure
        (including Neo4j unreachable / auth failure)
      - validate query_type is in QUERY_TYPES (error message should list valid types)
      - validate required params are present and are strings
      - run parameterized Cypher only (NEVER build Cypher with string formatting of user input)
      - return rows as a list of plain dicts (JSON-serializable)
      - return an ok result with an empty list when nothing matches (not an error)
    """
    raise NotImplementedError("TODO (Member 4)")


if __name__ == "__main__":
    # TODO (Member 4): example call for running the tool independently:
    #   python -m src.tools.graph_tool
    print(query_graph("prerequisites", {"course": "Machine Learning"}))