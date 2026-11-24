# Architecture

Owner: Member 5 (feature/agent-core). Fill in details as the agent is implemented.

## Overview

"Student Performance Advisor" is one LLM agent built with LangGraph. In each iteration the LLM analyzes the request and the previous tool results, then picks one action: call a tool, ask the user for missing info, or give the final answer. The loop may stop early and never exceeds **5 iterations**.

Three independent tools:

| Tool | Backend | Purpose |
|------|---------|---------|
| `predict_performance` | scikit-learn model (`models/`) | predict academic performance; what-if scenarios |
| `search_documents` | ChromaDB (`chroma_db/`) | semantic search over study guides |
| `query_graph` | Neo4j | courses, subjects, topics, prerequisites, resources |

## Diagram

```mermaid
flowchart TD
    U([User]) --> L[LLM node<br/>iteration += 1]
    L --> C{Conditional edge}
    C -- "tool call AND iteration < 5" --> T[Tool node]
    T --> ML[ML tool<br/>predict_performance]
    T --> CH[Chroma tool<br/>search_documents]
    T --> N4[Neo4j tool<br/>query_graph]
    ML --> L
    CH --> L
    N4 --> L
    C -- "final answer / ask user for info" --> F([Final answer])
    C -- "iteration limit reached" --> X[Final answer:<br/>what was found + what is unresolved]
    X --> F
```

## State

`AgentState` = `messages`, `tool_results`, `iteration` (see `src/agent/state.py` and `docs/contracts.md`).

## Loop rules

- TODO (Member 5): exact stop conditions and how the iteration limit produces a final answer.
- TODO (Member 5): how "ask user for missing info" ends a turn and resumes on the next user message.

## Error handling

- Tools never raise; they return `{"status": "error", "error": "..."}`.
- `tool_node` additionally catches unexpected exceptions and converts them to the same error dict.
- TODO (Member 5): how the LLM is told about errors and how repeated failures are handled.

## Logging

One structured entry per iteration (see `src/agent/logger.py`): iteration, optional reasoning, action, tool input, tool output.

## Module map

| Module | Responsibility |
|--------|----------------|
| `src/agent/state.py` | `AgentState` |
| `src/agent/nodes.py` | `llm_node`, `tool_node` |
| `src/agent/graph.py` | graph, conditional edges, iteration limit |
| `src/agent/prompts.py` | system prompt, tool descriptions |
| `src/agent/logger.py` | logging |
| `src/tools/__init__.py` | tool registry (LangChain tools) |
| `src/app/cli.py` | CLI |