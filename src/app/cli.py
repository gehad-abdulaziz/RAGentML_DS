"""Simple CLI chat loop.

Owner: Member 6 (feature/agent-interface).

Run:  python -m src.app.cli
"""
from __future__ import annotations


def main() -> None:
    """Interactive chat loop.

    TODO:
      - print a welcome message including the "educational advice only" disclaimer
      - loop: read user input; "exit"/"quit" ends the program; empty input is ignored
      - call src.agent.graph.run_agent(user_message) and print the answer
      - keep conversation context so the user can answer a "missing info" question
        (decide with Member 5 how history is passed between turns)
      - handle KeyboardInterrupt and unexpected errors without crashing
      - optional flag --mock to use mock tools (see src/tools/mock_tools.py)
    """
    raise NotImplementedError("TODO (Member 6)")


if __name__ == "__main__":
    main()