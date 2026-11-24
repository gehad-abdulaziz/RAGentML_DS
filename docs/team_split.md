# Team Split

TODO: replace "Member N" with names.

| Member | Name | Role | Branch | Files owned | Merges into |
|--------|------|------|--------|-------------|-------------|
| 1 | TODO | ML tool | `feature/ml-tool` | `src/ml/`, `src/tools/ml_tool.py`, `models/`, `data/raw/`, `data/processed/`, `tests/test_ml_tool.py` | `develop` |
| 2 | TODO | RAG ingestion | `feature/rag-ingestion` | `data/documents/`, `src/vectordb/build_index.py` | `develop` |
| 3 | TODO | RAG retrieval | `feature/rag-retrieval` | `src/tools/vector_tool.py`, `tests/test_vector_tool.py` | `develop` |
| 4 | TODO | Graph tool | `feature/graph-tool` | `src/graphdb/`, `src/tools/graph_tool.py`, `data/graph_seed/`, `docker-compose.yml`, `tests/test_graph_tool.py` | `develop` |
| 5 | TODO | Agent core (LangGraph) | `feature/agent-core` | `src/agent/state.py`, `src/agent/nodes.py`, `src/agent/graph.py`, `docs/architecture.md` | `develop` |
| 6 | TODO | Agent interface | `feature/agent-interface` | `src/tools/__init__.py`, `src/tools/mock_tools.py`, `src/agent/prompts.py`, `src/agent/logger.py`, `src/app/cli.py`, `src/config.py`, `tests/test_agent.py`, `tests/scenarios.md`, `logs/` | `develop` |

Shared files (each member edits only the section for their own part): `requirements.txt`, `README.md`, `docs/contracts.md`, `docs/data_sources.md`.

## Dependencies

- Members 1, 2, 4, 5 start in parallel from day 1.
- Member 3 starts with a small sample collection (3-4 documents) and switches to the full index when Member 2 finishes.
- Members 5 and 6 use `src/tools/mock_tools.py` until the real tools are merged. Member 6 depends on the `AgentState` shape from Member 5.
- Members 2 and 4 must agree on topic names so documents and graph data are consistent.
- Final integration and end-to-end tests (including error injection and iteration exhaustion) happen on `develop` after all tool branches are merged. Each member saves test logs for their own part in `logs/`.

## Everyone must be able to explain

Their own code and the overall flow (User -> LLM node -> conditional edge -> Tool node -> LLM, max 5 iterations -> final answer) for the project discussion.