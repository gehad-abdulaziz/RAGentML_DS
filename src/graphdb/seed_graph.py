"""Seed Neo4j from the CSV files in data/graph_seed/.

Owner: Member 4 (feature/graph-tool).

Run:  python -m src.graphdb.seed_graph            (add data / update existing)
      python -m src.graphdb.seed_graph --reset    (DELETE the whole graph first, then re-seed)

Requires Neo4j running (docker compose up -d) and NEO4J_* values in .env.

The script is IDEMPOTENT: it uses MERGE, so running it twice does not create duplicates.
All Cypher is parameterized (values are passed as parameters, never pasted into the text).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from neo4j import GraphDatabase

from src import config

SCHEMA_FILE = Path(__file__).with_name("schema.cypher")

# Every CSV file we read, and the columns we expect in it.
SEED_FILES = {
    "subjects": ["name"],
    "courses": ["name", "code", "subject"],
    "topics": ["name", "description"],
    "strategies": ["name", "description"],
    "resources": ["title", "url", "type"],
    "course_topics": ["course", "topic"],
    "course_prerequisites": ["prerequisite", "course"],
    "topic_requires": ["topic", "requires"],
    "topic_strategies": ["topic", "strategy"],
    "resource_topics": ["resource", "topic"],
}


def get_driver():
    """Create a neo4j driver from src.config (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)."""
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )
    # Fail early with a clear message if Neo4j is down or the password is wrong.
    driver.verify_connectivity()
    return driver


def apply_schema(driver) -> None:
    """Run the statements in src/graphdb/schema.cypher (uniqueness constraints)."""
    text = SCHEMA_FILE.read_text(encoding="utf-8")
    # Drop comment lines, then split the rest into statements at each semicolon.
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("//")]
    statements = [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
    with driver.session() as session:
        for statement in statements:
            session.run(statement)
    print(f"Schema applied ({len(statements)} statements).")


def load_seed_files() -> dict[str, list[dict]]:
    """Read every CSV in data/graph_seed/ and return {file_name: [row_dict, ...]}."""
    data: dict[str, list[dict]] = {}
    for name, columns in SEED_FILES.items():
        path = config.GRAPH_SEED_DIR / f"{name}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in columns if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path.name}: missing column(s) {missing}")
            # strip spaces so "Algebra " and "Algebra" are the same value
            data[name] = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    return data


def validate(data: dict[str, list[dict]]) -> None:
    """Check that every relationship points to a node that exists.

    Without this, a typo in a CSV (e.g. "Calculus 1" vs "Calculus I") would make Cypher's
    MATCH find nothing and the relationship would be SILENTLY skipped.
    """
    names = {
        "course": {r["name"] for r in data["courses"]},
        "subject": {r["name"] for r in data["subjects"]},
        "topic": {r["name"] for r in data["topics"]},
        "strategy": {r["name"] for r in data["strategies"]},
        "resource": {r["title"] for r in data["resources"]},
    }
    # (file, column, kind of node the column must refer to)
    checks = [
        ("courses", "subject", "subject"),
        ("course_topics", "course", "course"),
        ("course_topics", "topic", "topic"),
        ("course_prerequisites", "prerequisite", "course"),
        ("course_prerequisites", "course", "course"),
        ("topic_requires", "topic", "topic"),
        ("topic_requires", "requires", "topic"),
        ("topic_strategies", "topic", "topic"),
        ("topic_strategies", "strategy", "strategy"),
        ("resource_topics", "resource", "resource"),
        ("resource_topics", "topic", "topic"),
    ]
    problems = []
    for file_name, column, kind in checks:
        for row in data[file_name]:
            if row[column] not in names[kind]:
                problems.append(f"{file_name}.csv: {column}='{row[column]}' is not a known {kind}")
    if problems:
        raise ValueError("Seed data has broken references:\n  " + "\n  ".join(problems))


def seed(driver, data: dict[str, list[dict]]) -> None:
    """Create nodes first, then relationships, using parameterized MERGE statements.

    UNWIND $rows AS row  -> runs the statement once for every row in the list we pass in.
    MERGE                -> "create if it does not exist yet", which makes the script re-runnable.
    """
    node_queries = [
        ("subjects", "UNWIND $rows AS row MERGE (:Subject {name: row.name})"),
        (
            "courses",
            "UNWIND $rows AS row MERGE (c:Course {name: row.name}) SET c.code = row.code",
        ),
        (
            "topics",
            "UNWIND $rows AS row MERGE (t:Topic {name: row.name}) SET t.description = row.description",
        ),
        (
            "strategies",
            "UNWIND $rows AS row MERGE (s:Strategy {name: row.name}) SET s.description = row.description",
        ),
        (
            "resources",
            "UNWIND $rows AS row MERGE (r:Resource {title: row.title}) "
            "SET r.url = row.url, r.type = row.type",
        ),
    ]
    relationship_queries = [
        (
            "courses",  # Course -> Subject (the subject column lives in courses.csv)
            "UNWIND $rows AS row "
            "MATCH (c:Course {name: row.name}) MATCH (s:Subject {name: row.subject}) "
            "MERGE (c)-[:BELONGS_TO]->(s)",
        ),
        (
            "course_topics",
            "UNWIND $rows AS row "
            "MATCH (c:Course {name: row.course}) MATCH (t:Topic {name: row.topic}) "
            "MERGE (c)-[:HAS_TOPIC]->(t)",
        ),
        (
            "course_prerequisites",  # prerequisite is taken BEFORE course
            "UNWIND $rows AS row "
            "MATCH (p:Course {name: row.prerequisite}) MATCH (c:Course {name: row.course}) "
            "MERGE (p)-[:PREREQUISITE_OF]->(c)",
        ),
        (
            "topic_requires",  # topic needs `requires` first
            "UNWIND $rows AS row "
            "MATCH (t:Topic {name: row.topic}) MATCH (r:Topic {name: row.requires}) "
            "MERGE (t)-[:REQUIRES]->(r)",
        ),
        (
            "topic_strategies",
            "UNWIND $rows AS row "
            "MATCH (t:Topic {name: row.topic}) MATCH (s:Strategy {name: row.strategy}) "
            "MERGE (t)-[:BEST_STUDIED_WITH]->(s)",
        ),
        (
            "resource_topics",
            "UNWIND $rows AS row "
            "MATCH (r:Resource {title: row.resource}) MATCH (t:Topic {name: row.topic}) "
            "MERGE (r)-[:COVERS]->(t)",
        ),
    ]
    with driver.session() as session:
        for file_name, query in node_queries + relationship_queries:
            session.run(query, rows=data[file_name])


def reset_graph(driver) -> None:
    """Delete every node and relationship (constraints are kept).

    Needed after you REMOVE something from the CSV files: MERGE only adds/updates, it never deletes.
    """
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Graph cleared.")


def print_counts(driver) -> None:
    """Print how many nodes and relationships exist, so you can check the seed worked."""
    with driver.session() as session:
        print("\nNodes per label:")
        for rec in session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS n ORDER BY label"
        ):
            print(f"  {rec['label']:<10} {rec['n']}")
        print("Relationships per type:")
        for rec in session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS n ORDER BY type"
        ):
            print(f"  {rec['type']:<18} {rec['n']}")


def main() -> None:
    """driver -> load + validate CSVs -> (optional reset) -> apply_schema -> seed -> print counts."""
    driver = get_driver()
    try:
        data = load_seed_files()
        validate(data)  # check the CSVs BEFORE touching the database
        if "--reset" in sys.argv:
            reset_graph(driver)
        apply_schema(driver)
        seed(driver, data)
        print_counts(driver)
        print("\nSeed finished.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
