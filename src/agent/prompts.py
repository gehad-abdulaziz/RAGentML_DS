"""System prompt and tool descriptions.

Owner: Member 6 (feature/agent-interface).
"""
from __future__ import annotations

# TODO (Member 6): write the system prompt. It must tell the LLM to:
#   - act as an EDUCATIONAL study advisor only (no medical, legal, or financial advice)
#   - decide each iteration: call a tool, ask the user for missing info, or give the final answer
#   - stop early when it has enough information; maximum 5 iterations
#   - for what-if scenarios (e.g., changing study hours), call the ML tool multiple times
#   - ground the final answer ONLY in tool results; never invent values, sources or links
#   - if a tool returns {"status": "error"}, explain what failed and try a fix or ask the user
#   - if iterations run out, state what was found and what is unresolved
SYSTEM_PROMPT: str = ""  # TODO

# TODO (Member 6): tool descriptions shown to the LLM (when to use, inputs, example calls).
ML_TOOL_DESCRIPTION: str = ""  # TODO: predict_performance(features: dict)
VECTOR_TOOL_DESCRIPTION: str = ""  # TODO: search_documents(query: str, k: int = 3)
GRAPH_TOOL_DESCRIPTION: str = ""  # TODO: query_graph(query_type: str, params: dict) + valid query types

# TODO (Member 6): extra prompt used when the iteration limit is reached
MAX_ITERATIONS_PROMPT: str = ""  # TODO: "Give a final answer: what was found, what is unresolved."