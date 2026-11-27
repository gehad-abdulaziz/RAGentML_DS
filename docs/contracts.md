# Tool Contracts

Shared document. Each member edits only the section for their own tool.

## General rules (all tools)

Every tool returns a plain, JSON-serializable `dict`:

```python
# success
{"status": "ok", "data": {...}}
# failure
{"status": "error", "error": "<clear message>"}
```

- Tools must **NEVER raise** to the agent. Wrap the body in `try/except` and return the error dict.
- Every tool must work **independently** (callable from a Python shell / `python -m` without the agent).
- Error messages must be clear enough for the LLM to explain or fix (e.g., name the missing field or list valid values).
- Tool results are passed to the LLM as structured JSON.

## 1. ML tool — `predict_performance` (Member 1)

File: `src/tools/ml_tool.py`

```python
predict_performance(features: dict) -> dict
```

Success:

```json
{"status": "ok",
 "data": {"prediction": 72.4, "unit": "exam score (0-100)", "inputs_used": {"Hours_Studied": 20, "Attendance": 90, "Sleep_Hours": 7}}}
```

- Validates required features and value ranges (TODO Member 1: list them below).
- Failure examples: missing feature, value out of range, wrong type, model file not found.

| Feature | Type | Allowed range | Required |
|---------|------|---------------|----------|
| TODO | TODO | TODO | TODO |

## 2. Vector tool — `search_documents` (Member 3)

File: `src/tools/vector_tool.py`

```python
search_documents(query: str, k: int = 3) -> dict
```

Success:

```json
{"status": "ok",
 "data": {"results": [{"text": "...", "source": "active_recall.md", "score": 0.82}]}}
```

- `score`: higher = more relevant (TODO Member 3: document how it is computed).
- `source` must come from the chunk metadata written by `build_index.py` (never invented).
- Failure examples: empty query, invalid `k`, index missing/empty.

## 3. Graph tool — `query_graph` (Member 4)

File: `src/tools/graph_tool.py`

```python
query_graph(query_type: str, params: dict) -> dict
```

Success:

```json
{"status": "ok", "data": {"rows": [{"...": "..."}]}}
```

`query_type` is one of a fixed set. Cypher must be **parameterized** (no string formatting of user input).

| query_type | params | Returns (rows) |
|------------|--------|----------------|
| `prerequisites` | `{"course": str}` | `{course, prerequisite, distance}`; distance 1 = direct prerequisite, 2+ = indirect |
| `topics_in_course` | `{"course": str}` | `{course, topic, description}` |
| `resources_for_topic` | `{"topic": str}` | `{topic, title, url, type}` |
| `courses_in_subject` | `{"subject": str}` | `{subject, course, code}` |
| `strategies_for_topic` (extra) | `{"topic": str}` | `{topic, strategy, description}` |
| `topic_prerequisites` (extra) | `{"topic": str}` | `{topic, requires, distance}` |

Notes: matching is case-insensitive; surrounding spaces are ignored. Valid values are in `data/graph_seed/`
(courses e.g. "Calculus II", "Linear Algebra", "Data Structures and Algorithms"; subjects: "Mathematics", "Programming").

- Unknown entity -> `{"status": "ok", "data": {"rows": []}}` (not an error).
- Failure examples: unknown `query_type`, missing params, Neo4j unreachable.

### Graph schema (starting point)

```
(Course)-[:BELONGS_TO]->(Subject)
(Course)-[:HAS_TOPIC]->(Topic)
(Course)-[:PREREQUISITE_OF]->(Course)
(Topic)-[:REQUIRES]->(Topic)
(Resource)-[:COVERS]->(Topic)
(Topic)-[:BEST_STUDIED_WITH]->(Strategy)
```
Schema note: `A-[:PREREQUISITE_OF]->B` = take A before B.  `A-[:REQUIRES]->B` = learn B before A.

Also in `src/graphdb/schema.cypher`.

> **Consistency note:** topic names in the graph and in `data/documents/` must match. Members 2 and 4 agree on a shared topic list (TODO: link/list it here).

## Agent-side contract (Members 5 & 6)

`AgentState` (`src/agent/state.py`):

| Field | Type | Meaning |
|-------|------|---------|
| `messages` | list (add_messages) | conversation history |
| `tool_results` | list[dict] | `{"iteration", "tool", "input", "output"}` per tool call |
| `iteration` | int | iterations used so far; max `MAX_ITERATIONS = 5` |

Iteration log entry (one per iteration): `iteration`, optional `reasoning`, `action` (`call_tool` / `ask_user` / `final_answer`), `tool`, `input`, `output`.
