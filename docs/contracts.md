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
 "data": {"prediction": 72.4, "unit": "exam_score", "inputs_used": {"Hours_Studied": 20, "Attendance": 90, "Sleep_Hours": 7}}}
```

- Validates required features and value ranges.
- Failure examples: missing feature, value out of range, wrong type, model file not found.

| Feature | Type | Allowed range / Categories | Required |
|---------|------|---------------------------|----------|
| `Hours_Studied` | float / int | [0, 100] | Yes |
| `Attendance` | float / int | [0, 100] | Yes |
| `Sleep_Hours` | float / int | [0, 24] | Yes |
| `Previous_Scores` | float / int | [0, 100] | Yes |
| `Tutoring_Sessions` | float / int | [0, 20] | Yes |
| `Physical_Activity` | float / int | [0, 30] | Yes |
| `Parental_Involvement` | str | ["Low", "Medium", "High"] | Yes |
| `Access_to_Resources` | str | ["Low", "Medium", "High"] | Yes |
| `Motivation_Level` | str | ["Low", "Medium", "High"] | Yes |
| `Family_Income` | str | ["Low", "Medium", "High"] | Yes |
| `Teacher_Quality` | str | ["Low", "Medium", "High"] | Yes |
| `Peer_Influence` | str | ["Negative", "Neutral", "Positive"] | Yes |
| `Parental_Education_Level` | str | ["High School", "College", "Postgraduate"] | Yes |
| `Distance_from_Home` | str | ["Near", "Moderate", "Far"] | Yes |
| `Extracurricular_Activities` | str | ["Yes", "No"] | Yes |
| `Internet_Access` | str | ["Yes", "No"] | Yes |
| `School_Type` | str | ["Public", "Private"] | Yes |
| `Learning_Disabilities` | str | ["Yes", "No"] | Yes |
| `Gender` | str | ["Male", "Female"] | Yes |

## 2. Vector tool — `search_documents` (Member 3)

File: `src/tools/vector_tool.py`

```python
search_documents(query: str, k: int = 3) -> dict
```

Success:

```json
{"status": "ok",
 "data": {"results": [{"text": "...", "source": "06_Active_Recall_Spaced_Repetition_Guide.pdf",
                       "title": "Active Recall Spaced Repetition Guide", "page": 2,
                       "chunk_id": "06_Active_Recall_Spaced_Repetition_Guide_page_2_chunk_0",
                       "score": 0.8213}]}}
```

- `score`: cosine similarity between the query and the chunk, in [-1, 1]; **higher = more relevant**, `1.0` = same direction. Chroma stores cosine distance (`1 - similarity`); the tool converts it. Results are sorted best first.
- `source`, `title`, `page`, `chunk_id` come from the chunk metadata written by `build_index.py` (never invented). Chunks without a `source` are skipped.
- `k`: integer from 1 to 10 (`MAX_K`). Fewer than `k` results are returned when the index holds fewer chunks.
- `query`: non-empty string, at most 1000 characters after stripping spaces. English and Arabic both work.
- Failure examples: empty or non-string query, query too long, invalid `k`, no index at `CHROMA_PATH` (or empty collection), embedding model unavailable. Each error message says what to fix (e.g., `python -m src.vectordb.build_index`).

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
