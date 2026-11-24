"""Seed Neo4j from the CSV/JSON files in data/graph_seed/.

Owner: Member 4 (feature/graph-tool).

Run:  python -m src.graphdb.seed_graph

Requires Neo4j running (docker compose up -d) and NEO4J_* values in .env.
"""
from __future__ import annotations


def get_driver():
    """Create a neo4j driver from src.config (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).

    TODO: return neo4j.GraphDatabase.driver(...)
    """
    raise NotImplementedError


def apply_schema(driver) -> None:
    """Run the statements in src/graphdb/schema.cypher (constraints etc.).

    TODO: read the file, split into statements, execute each.
    """
    raise NotImplementedError


def load_seed_files() -> dict:
    """Read the CSV/JSON files from data/graph_seed/.

    TODO: return nodes and relationships ready to be written.
    """
    raise NotImplementedError


def seed(driver, data: dict) -> None:
    """Create nodes and relationships using parameterized, idempotent (MERGE) Cypher.

    TODO: nodes first, then relationships.
    """
    raise NotImplementedError


def main() -> None:
    """TODO: driver -> apply_schema -> load_seed_files -> seed -> print node/relationship counts."""
    raise NotImplementedError


if __name__ == "__main__":
    main()