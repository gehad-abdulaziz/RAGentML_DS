"""Interactive CLI: login -> choose session -> chat with memory.

Run:   python -m src.app.cli            (real tools)
       python -m src.app.cli --mock     (mock tools, no Neo4j / Chroma / model file needed)

In-chat commands:  /help  /whoami  /profile  /memory  /remember <text>  /forget <id>
                   /sessions  /switch <n>  /new  /logout  exit | quit
"""
from __future__ import annotations

import argparse
import getpass
import sys

from src import config
from src.agent.logger import log_event, setup_logger
from src.app.service import chat_turn
from src.memory import store
from src.memory.auth import AuthError, User, login, register
from src.memory.db import init_db

WELCOME = """
============================================================
  Student Performance Advisor
  Educational advice only - not a substitute for a teacher,
  doctor, or any professional. Predictions are estimates.
============================================================
"""

HELP = """Commands:
  /whoami            show who is logged in
  /profile           show saved study profile (used for score predictions)
  /memory            list long-term memories
  /remember <text>   save a fact, e.g. /remember I am weak in Calculus
  /forget <id>       delete one memory (id from /memory)
  /sessions          list your chats      /switch <n>  continue chat number n
  /new               start a new chat     /logout      switch user
  exit | quit        leave
"""


def _ask(prompt: str, secret: bool = False) -> str:
    if secret:
        try:
            return getpass.getpass(prompt)
        except (EOFError, KeyboardInterrupt):
            raise
        except Exception:  # noqa: BLE001 - no TTY (some IDE consoles)
            return input(prompt)
    return input(prompt)


def auth_flow(preset_username: str | None = None) -> User | None:
    """Login or register. Returns None if the user gives up."""
    print("1) Log in   2) Create account   q) Quit")
    choice = "1" if preset_username else _ask("> ").strip().lower()
    if choice in {"q", "quit", "exit"}:
        return None
    for _ in range(3):
        try:
            if choice == "2":
                username = _ask("Choose a username: ").strip()
                display = _ask("Your name (shown in chat, Enter = same): ").strip() or None
                password = _ask("Choose a password: ", secret=True)
                if password != _ask("Repeat password: ", secret=True):
                    print("Passwords do not match.")
                    continue
                user = register(username, password, display)
                log_event("register", user_id=user.id)
                return user
            username = preset_username or _ask("Username: ").strip()
            user = login(username, _ask("Password: ", secret=True))
            log_event("login", user_id=user.id)
            return user
        except AuthError as exc:
            print(f"! {exc}")
            preset_username = None
    print("Too many tries.")
    return None


def pick_session(user: User, force_new: bool) -> str:
    sessions = store.list_sessions(user.id, limit=5)
    if force_new or not sessions:
        return store.create_session(user.id)
    print("Your recent chats:")
    for i, s in enumerate(sessions, 1):
        print(f"  {i}) {s['title']}  ({s['n_messages']} messages)")
    answer = input("Enter = continue 1, n = new chat, or a number: ").strip().lower()
    if answer == "n":
        return store.create_session(user.id)
    if answer.isdigit() and 1 <= int(answer) <= len(sessions):
        return sessions[int(answer) - 1]["id"]
    return sessions[0]["id"]


def _greet(user: User, session_id: str) -> None:
    history = store.get_history(user.id, session_id, 2)
    n_mem = len(store.list_memories(user.id))
    profile = store.get_profile(user.id)
    print(f"\nWelcome back, {user.display_name}!" if history else f"\nHello, {user.display_name}!")
    print(f"Logged in as '{user.username}'. Saved memories: {n_mem}, profile values: {len(profile)}.")
    if history:
        last_user = next((c for r, c in reversed(history) if r == "user"), None)
        if last_user:
            print(f"Last time you asked: \"{last_user[:80]}\"")
    print("Type /help for commands.\n")


def _handle_command(cmd: str, user: User, session_id: str, sessions_cache: list) -> tuple[str, str | None]:
    """Return (action, new_session_id). action: 'continue' | 'logout' | 'quit'."""
    parts = cmd.split(maxsplit=1)
    name, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
    if name in {"/exit", "/quit"}:
        return "quit", None
    if name == "/help":
        print(HELP)
    elif name == "/whoami":
        print(f"{user.display_name} ({user.username}), chat {session_id[:8]}")
    elif name == "/profile":
        profile = store.get_profile(user.id)
        print("\n".join(f"  {k} = {v}" for k, v in profile.items()) or "  (empty)")
    elif name == "/memory":
        mems = store.list_memories(user.id)
        print("\n".join(f"  [{m['id']}] {m['text']}" for m in mems) or "  (no memories yet)")
    elif name == "/remember":
        ok, msg = store.add_memory(user.id, arg) if arg else (False, "Usage: /remember <text>")
        print(msg)
    elif name == "/forget":
        if arg.isdigit() and store.delete_memory(user.id, int(arg)):
            print("Deleted.")
        else:
            print("No such memory id. Use /memory to see ids.")
    elif name == "/sessions":
        sessions_cache[:] = store.list_sessions(user.id, 10)
        for i, s in enumerate(sessions_cache, 1):
            mark = "*" if s["id"] == session_id else " "
            print(f" {mark}{i}) {s['title']}  ({s['n_messages']} messages)")
    elif name == "/switch":
        if arg.isdigit() and 1 <= int(arg) <= len(sessions_cache):
            sid = sessions_cache[int(arg) - 1]["id"]
            print("Switched.")
            return "continue", sid
        print("Run /sessions first, then /switch <number>.")
    elif name == "/new":
        print("Started a new chat.")
        return "continue", store.create_session(user.id)
    elif name == "/logout":
        return "logout", None
    else:
        print("Unknown command. Type /help.")
    return "continue", None


def chat_loop(user: User, session_id: str) -> str:
    """Return 'logout' or 'quit'."""
    sessions_cache: list = []
    while True:
        try:
            raw = input(f"{user.display_name}> ")
        except EOFError:
            return "quit"
        except KeyboardInterrupt:
            print("\n(Ctrl+C) Bye!")
            return "quit"

        text = raw.strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            return "quit"
        if text.startswith("/"):
            action, new_sid = _handle_command(text, user, session_id, sessions_cache)
            if action != "continue":
                return action
            if new_sid:
                session_id = new_sid
                _greet(user, session_id)
            continue

        try:
            result = chat_turn(user, session_id, text)
            print(f"\nAdvisor> {result.answer}\n")
            if result.saved_profile:
                print(f"  (saved to your profile: {', '.join(result.saved_profile)})\n")
        except KeyboardInterrupt:
            print("\n(cancelled this question)\n")
        except Exception as exc:  # noqa: BLE001 - the chat must survive any error
            log_event("cli_error", user_id=user.id, error=f"{type(exc).__name__}: {exc}"[:300])
            print(f"\nSomething went wrong ({type(exc).__name__}). Please try again.\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Student Performance Advisor (CLI)")
    parser.add_argument("--mock", action="store_true", help="use mock tools (no Neo4j/Chroma/model needed)")
    parser.add_argument("--username", help="log in directly as this user")
    parser.add_argument("--new-chat", action="store_true", help="always start a new chat")
    parser.add_argument("--quiet-logs", action="store_true", help="do not print iteration logs to the console")
    args = parser.parse_args(argv)

    setup_logger(console=not args.quiet_logs)
    if args.mock:
        from src.tools import set_mock_mode
        set_mock_mode(True)
    init_db()

    print(WELCOME)
    print(f"LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL or 'default'}   tools: "
          f"{'MOCK' if args.mock or config.USE_MOCK_TOOLS else 'real'}   log: {config.AGENT_LOG_FILE}")

    try:
        while True:
            user = auth_flow(args.username)
            if user is None:
                break
            args.username = None
            session_id = pick_session(user, args.new_chat)
            _greet(user, session_id)
            if chat_loop(user, session_id) == "quit":
                break
    except (KeyboardInterrupt, EOFError):
        print()
    print("Goodbye!")


if __name__ == "__main__":
    sys.exit(main())
