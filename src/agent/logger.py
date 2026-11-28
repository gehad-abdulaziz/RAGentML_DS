"""Per-iteration structured logging (console + JSON lines file in logs/).

Owner: Member 6 (feature/agent-interface).

Every iteration logs: iteration number, selected action, optional reasoning,
tool name, tool input, tool output.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src import config

_current_log_file: Path | None = None
_logger_initialized = False


def _sanitize(obj: Any) -> Any:
    """Mask sensitive fields such as API keys and secrets."""
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            if any(secret_term in k.lower() for secret_term in ("key", "secret", "password", "token")):
                sanitized[k] = "***REDACTED***"
            else:
                sanitized[k] = _sanitize(v)
        return sanitized
    elif isinstance(obj, list):
        return [_sanitize(item) for item in obj]
    return obj


def setup_logger(log_file: str | Path | None = None) -> Path:
    """Configure console + file logging.

    Writes to logs/agent.log (or given path). Never logs API keys.
    """
    global _current_log_file, _logger_initialized

    if log_file:
        target_path = Path(log_file)
        if not target_path.is_absolute():
            target_path = config.LOGS_DIR / target_path
    else:
        target_path = config.LOGS_DIR / config.LOG_FILE_NAME

    target_path.parent.mkdir(parents=True, exist_ok=True)
    _current_log_file = target_path

    # Configure standard library logging handler if not already done
    root_logger = logging.getLogger("ragentml")
    root_logger.setLevel(logging.INFO)

    if not _logger_initialized:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        _logger_initialized = True

    return _current_log_file


def get_current_log_file() -> Path:
    """Return the active log file path, initializing default if needed."""
    global _current_log_file
    if _current_log_file is None:
        return setup_logger()
    return _current_log_file


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
    """
    log_file = get_current_log_file()

    entry: dict[str, Any] = {
        "iteration": iteration,
        "action": action,
    }

    if reasoning:
        entry["reasoning"] = str(reasoning).strip()
    if tool_name:
        entry["tool"] = tool_name
    if tool_input is not None:
        entry["input"] = _sanitize(tool_input)
    if tool_output is not None:
        entry["output"] = _sanitize(tool_output)

    # 1. Console output: clean, readable line
    console_parts = [f"[Agent] Iteration {iteration} | Action: {action}"]
    if tool_name:
        console_parts.append(f"| Tool: {tool_name}")
    if reasoning:
        short_reason = reasoning[:100] + ("..." if len(reasoning) > 100 else "")
        console_parts.append(f"| Reason: {short_reason}")
    print(" ".join(console_parts))

    # 2. Append JSON line to log file
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[Warning] Failed to write iteration log to {log_file}: {exc}")
