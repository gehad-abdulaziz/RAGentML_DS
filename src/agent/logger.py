"""Per-iteration structured logging (console + file in logs/).

Owner: Member 6 (feature/agent-interface).

Every iteration must log: iteration number, optional short reasoning, selected action,
tool input, tool output.
"""
from __future__ import annotations


def setup_logger(log_file: str | None = None):
    """Configure console + file logging.

    TODO: write to logs/agent.log (or the given file); do NOT log API keys.
    """
    raise NotImplementedError("TODO (Member 6)")


def log_iteration(
    iteration: int,
    action: str,
    reasoning: str | None = None,
    tool_name: str | None = None,
    tool_input: dict | None = None,
    tool_output: dict | None = None,
) -> None:
    """Log one agent iteration as structured data.

    action: "call_tool" | "ask_user" | "final_answer"

    TODO:
      - print a readable line to the console
      - append one JSON line per iteration to the log file, e.g.
        {"iteration": 1, "reasoning": "...", "action": "call_tool",
         "tool": "predict_performance", "input": {...}, "output": {...}}
      - document the final format in README.md (Agent behavior -> Logging format)
    """
    raise NotImplementedError("TODO (Member 6)")