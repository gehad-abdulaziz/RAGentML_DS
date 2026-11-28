"""Per-iteration structured logging (console + file in logs/).

Every iteration logs: iteration number, short reasoning (if the model gave any),
selected action, tool input, tool output.  One JSON object per line is appended to
`logs/agent.log` (UTF-8) and a readable line is printed to the console.

Log line format (JSON Lines):
    {"ts": "2026-11-28T12:00:00", "event": "iteration", "iteration": 1,
     "action": "call_tool", "reasoning": "...", "tool": "query_graph",
     "input": {...}, "output": {...}, "user_id": 3, "session_id": "ab12..."}

action is one of: "call_tool" | "ask_user" | "final_answer" | "blocked".
API keys, passwords, e-mails and phone numbers are redacted before writing.
Logging never raises: a broken log must not break the agent.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src import config

_FILE_LOGGER = "agent.file"
_CONSOLE_LOGGER = "agent.console"
_state: dict[str, Any] = {"ready": False, "file": None}

MAX_FILE_FIELD_CHARS = 20000
MAX_CONSOLE_FIELD_CHARS = 300


def _redact(text: str) -> str:
    from src.agent.middleware import redact_sensitive  # local import: avoid cycles

    return redact_sensitive(text)


def setup_logger(log_file: str | Path | None = None, console: bool = True,
                 level: int = logging.INFO) -> None:
    """Configure console + file logging (safe to call many times)."""
    path = Path(log_file) if log_file else Path(config.AGENT_LOG_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_logger = logging.getLogger(_FILE_LOGGER)
    file_logger.setLevel(level)
    file_logger.propagate = False
    for handler in list(file_logger.handlers):
        handler.close()
        file_logger.removeHandler(handler)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    file_logger.addHandler(handler)

    console_logger = logging.getLogger(_CONSOLE_LOGGER)
    console_logger.setLevel(level)
    console_logger.propagate = False
    for h in list(console_logger.handlers):
        console_logger.removeHandler(h)
    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(message)s"))
        console_logger.addHandler(stream)

    _state["ready"] = True
    _state["file"] = str(path)


def _ensure() -> None:
    if not _state["ready"] or _state["file"] != str(config.AGENT_LOG_FILE):
        # first use (or tests changed the path): console off, file on
        setup_logger(console=False)


def _cut(value: Any, limit: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    if len(text) <= limit:
        return value
    return text[:limit] + "...[truncated]"


def _write(record: dict[str, Any], console_line: str) -> None:
    try:
        _ensure()
        line = _redact(json.dumps(record, ensure_ascii=False, default=str))
        logging.getLogger(_FILE_LOGGER).info(line)
        logging.getLogger(_CONSOLE_LOGGER).info(_redact(console_line))
    except Exception:  # noqa: BLE001 - logging must never break the agent
        pass


def log_iteration(
    iteration: int,
    action: str,
    reasoning: str | None = None,
    tool_name: str | None = None,
    tool_input: Any = None,
    tool_output: Any = None,
    user_id: Any = None,
    session_id: str | None = None,
) -> None:
    """Log one agent iteration as structured data (console + logs/agent.log)."""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": "iteration",
        "iteration": iteration,
        "action": action,
        "reasoning": _cut(reasoning, 1000) if reasoning else None,
        "tool": tool_name,
        "input": _cut(tool_input, MAX_FILE_FIELD_CHARS),
        "output": _cut(tool_output, MAX_FILE_FIELD_CHARS),
        "user_id": user_id,
        "session_id": session_id,
    }
    parts = [f"[iter {iteration}] {action}"]
    if tool_name:
        parts.append(f"tool={tool_name}")
    if tool_input is not None:
        parts.append(f"input={_cut(tool_input, MAX_CONSOLE_FIELD_CHARS)}")
    if tool_output is not None:
        parts.append(f"output={_cut(tool_output, MAX_CONSOLE_FIELD_CHARS)}")
    if reasoning:
        parts.append(f"reasoning={_cut(reasoning, 160)}")
    _write(record, " | ".join(str(p) for p in parts))


def log_event(event: str, user_id: Any = None, session_id: str | None = None, **fields: Any) -> None:
    """Log a non-iteration event (guardrail hit, login, memory write, error ...)."""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "user_id": user_id,
        "session_id": session_id,
        **fields,
    }
    detail = " ".join(f"{k}={_cut(v, 120)}" for k, v in fields.items())
    _write(record, f"[{event}] {detail}")
