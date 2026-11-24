# Contributing

## Branch rules

- `main` is protected: final versions only, merged from `develop` at milestones.
- `develop` is the integration branch.
- Every feature branch is created from `develop` and opens a **Pull Request into `develop`**.
- Every PR needs **one review from another member** before merging.
- **Never commit `.env` or API keys.** Use `.env.example` for placeholders only.
- Edit only the files you own. In shared files (`requirements.txt`, `README.md`, `docs/contracts.md`, `docs/data_sources.md`) edit only the section for your own part.

## Branch table

| Branch | Owner | Files owned | Merges into |
|--------|-------|-------------|-------------|
| `main` | — (protected) | everything, final only | — |
| `develop` | — (integration) | everything, integration | `main` (at milestones) |
| `feature/ml-tool` | Member 1 | `src/ml/`, `src/tools/ml_tool.py`, `models/`, `data/raw/`, `data/processed/`, `tests/test_ml_tool.py` | `develop` |
| `feature/rag-ingestion` | Member 2 | `data/documents/`, `src/vectordb/build_index.py` | `develop` |
| `feature/rag-retrieval` | Member 3 | `src/tools/vector_tool.py`, `tests/test_vector_tool.py` | `develop` |
| `feature/graph-tool` | Member 4 | `src/graphdb/`, `src/tools/graph_tool.py`, `data/graph_seed/`, `docker-compose.yml`, `tests/test_graph_tool.py` | `develop` |
| `feature/agent-core` | Member 5 | `src/agent/state.py`, `src/agent/nodes.py`, `src/agent/graph.py`, `docs/architecture.md` | `develop` |
| `feature/agent-interface` | Member 6 | `src/tools/__init__.py`, `src/tools/mock_tools.py`, `src/agent/prompts.py`, `src/agent/logger.py`, `src/app/cli.py`, `src/config.py`, `tests/test_agent.py`, `tests/scenarios.md`, `logs/` | `develop` |

## Daily workflow

```bash
git checkout feature/<your-branch>
git pull origin develop          # keep up to date with integration
# ... work, commit small and often ...
git add <your files>
git commit -m "ml-tool: add input validation"
git push origin feature/<your-branch>
# open a Pull Request: feature/<your-branch> -> develop, request one reviewer
```

Commit message style: `<area>: <what changed>` (e.g., `graph-tool: add prerequisites query`).

## Pull request checklist

- [ ] Only files I own (or my section of shared files) are changed
- [ ] Follows `docs/contracts.md` (return shape, tools never raise)
- [ ] My tests run and a test log is saved in `logs/`
- [ ] No `.env`, API keys or large raw data committed
- [ ] One teammate reviewed it

## Dependencies between members

- Members 1, 2, 4, 5 start in parallel from day 1.
- Member 3 starts with a small sample collection (3-4 documents) and switches to the full index when Member 2 finishes.
- Members 5 and 6 use `src/tools/mock_tools.py` until the real tools are merged; Member 6 depends on the `AgentState` shape from Member 5.
- Members 2 and 4 must agree on topic names so documents and graph data are consistent.
- Final integration and end-to-end tests (error injection, iteration exhaustion) happen on `develop` after all tool branches are merged. Each member saves test logs for their own part in `logs/`.

## Merge conflicts

If a conflict appears in a file you do not own, ask the owner before resolving it.