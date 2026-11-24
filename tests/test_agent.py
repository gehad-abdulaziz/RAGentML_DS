"""End-to-end agent tests. Owner: Member 6 (feature/agent-interface).

Initially run with mock tools (src/tools/mock_tools.py); after all tool branches are merged
into develop, run with the real tools. Scenarios are described in tests/scenarios.md.
Save the output/logs to logs/.
"""
import pytest  # noqa: F401


def test_early_stop():
    """TODO: a simple question is answered in fewer than 5 iterations."""
    pytest.skip("TODO (Member 6)")


def test_what_if_calls_ml_tool_multiple_times():
    """TODO: 'what if I study 10 more hours?' -> ML tool called at least twice."""
    pytest.skip("TODO (Member 6)")


def test_missing_info_asks_user():
    """TODO: 'predict my score' without features -> agent asks for the missing info (no tool crash)."""
    pytest.skip("TODO (Member 6)")


def test_tool_error_does_not_crash():
    """TODO: force a tool error (mock error switch / monkeypatch) -> agent handles it and answers."""
    pytest.skip("TODO (Member 6)")


def test_invalid_input_handled():
    """TODO: invalid values (e.g., negative study hours) -> handled without crashing."""
    pytest.skip("TODO (Member 6)")


def test_iteration_exhaustion_gives_final_answer():
    """TODO: force the LLM to keep calling tools -> loop stops at 5 and the final answer says
    what was found and what is unresolved."""
    pytest.skip("TODO (Member 6)")


def test_iteration_never_exceeds_max():
    """TODO: assert final state iteration <= MAX_ITERATIONS in every scenario."""
    pytest.skip("TODO (Member 6)")


def test_final_answer_grounded_in_tool_results():
    """TODO: values/sources in the answer appear in tool outputs (nothing invented)."""
    pytest.skip("TODO (Member 6)")