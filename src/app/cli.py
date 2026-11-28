"""Interactive CLI chat loop for Student Performance Advisor.

Owner: Member 6 (feature/agent-interface).

Run:
    python -m src.app.cli
    python -m src.app.cli --mock
"""
from __future__ import annotations

import argparse
import os
import sys

from langchain_core.messages import BaseMessage

from src import config
from src.agent.logger import setup_logger


def print_banner() -> None:
    """Print the welcome banner and educational disclaimer."""
    banner = """
================================================================================
🎓  Student Performance Advisor (AI Academic Advisor)
================================================================================
Disclaimer:
  This system provides educational planning and advisory guidance only.
  It is not a substitute for official university academic counseling or
  official grading assessments.

Type your questions below.
Commands:
  - 'exit' or 'quit' to leave
  - 'clear' to reset conversation history
================================================================================
"""
    print(banner)


def main() -> None:
    """Run the interactive chat loop with conversation history."""
    parser = argparse.ArgumentParser(description="Student Performance Advisor CLI")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock tools instead of live databases and ML models.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use mock LLM responses for offline testing.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Custom file path for iteration logs (default: logs/agent.log).",
    )
    args = parser.parse_args()

    if args.mock:
        config.USE_MOCK_TOOLS = True
        os.environ["USE_MOCK_TOOLS"] = "1"

    if args.mock_llm:
        config.LLM_PROVIDER = "mock"
        os.environ["LLM_PROVIDER"] = "mock"

    log_path = setup_logger(args.log_file)
    print_banner()
    print(f"[*] Logging active. Iteration details saved to: {log_path}")
    if config.USE_MOCK_TOOLS:
        print("[*] Mode: Running with MOCK tools enabled.")

    from src.agent.graph import run_agent

    conversation_history: list[BaseMessage] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! Wishing you academic success.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q", ":q"):
            print("Session ended. Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            print("[*] Conversation history cleared.")
            continue

        try:
            print("\n[Thinking...]")
            result_state = run_agent(
                input_data=user_input,
                previous_messages=conversation_history,
            )

            # Update conversation history with the full message chain
            conversation_history = result_state["messages"]

            # Print the final assistant message
            last_message = conversation_history[-1]
            content = getattr(last_message, "content", "")
            print(f"\nAdvisor:\n{content}")

        except Exception as exc:
            print(f"\n[Error] An error occurred while processing your request: {exc}")
            print("Please try rephrasing your question or check system logs.")


if __name__ == "__main__":
    main()
