"""End-to-end agent tests. Owner: Member 6 (feature/agent-interface).

Tests are deterministic and self-contained, using mock tools and scripted LLM
responses to verify the exact agent behaviors specified in the course PDF.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    import pytest
except ImportError:
    class _DummyPytest:
        def fixture(self, *args, **kwargs):
            return lambda fn: fn
    pytest = _DummyPytest()

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src import config
from src.agent.graph import agent_graph, run_agent
from src.agent.logger import setup_logger
from src.tools.mock_tools import set_mock_force_error


class ScriptedChatModel(BaseChatModel):
    """A deterministic test ChatModel returning pre-scripted AIMessages."""

    scripted_responses: List[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted_chat_model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.index < len(self.scripted_responses):
            resp = self.scripted_responses[self.index]
            self.index += 1
        else:
            resp = self.scripted_responses[-1]
        generation = ChatGeneration(message=resp)
        return ChatResult(generations=[generation])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Ensure tests run with mock tools and clean logging."""
    setup_logger("test_agent.log")
    monkeypatch.setattr(config, "USE_MOCK_TOOLS", True)
    set_mock_force_error(False)
    yield
    set_mock_force_error(False)


def test_early_stop(monkeypatch):
    """Scenario 1: A simple question is answered in fewer than 5 iterations."""
    # Turn 1: LLM calls search_documents
    # Turn 2: LLM summarizes the retrieved chunks and answers
    msg1 = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_documents",
                "args": {"query": "What is spaced repetition?", "k": 2},
                "id": "call_rag_1",
            }
        ],
    )
    msg2 = AIMessage(
        content="Spaced repetition involves reviewing material at increasing intervals to optimize long-term memory."
    )

    mock_llm = ScriptedChatModel(scripted_responses=[msg1, msg2])
    monkeypatch.setattr("src.agent.nodes.get_llm", lambda *a, **kw: mock_llm)

    state = run_agent("What is spaced repetition?")
    assert state["iteration"] < config.MAX_ITERATIONS
    assert state["iteration"] == 1
    assert len(state["tool_results"]) == 1
    assert state["tool_results"][0]["tool"] == "search_documents"
    assert "Spaced repetition" in state["messages"][-1].content


def test_what_if_calls_ml_tool_multiple_times(monkeypatch):
    """Scenario 3: 'What if I study 15 instead of 10 hours?' -> ML tool called multiple times."""
    call1 = {
        "name": "predict_performance",
        "args": {"features": {"Hours_Studied": 10, "Attendance": 85, "Sleep_Hours": 7}},
        "id": "call_ml_1",
    }
    call2 = {
        "name": "predict_performance",
        "args": {"features": {"Hours_Studied": 15, "Attendance": 85, "Sleep_Hours": 7}},
        "id": "call_ml_2",
    }

    msg1 = AIMessage(content="", tool_calls=[call1])
    msg2 = AIMessage(content="", tool_calls=[call2])
    msg3 = AIMessage(
        content="Studying 10 hours yields an estimated score of 72.1, whereas studying 15 hours increases it to 77.6."
    )

    mock_llm = ScriptedChatModel(scripted_responses=[msg1, msg2, msg3])
    monkeypatch.setattr("src.agent.nodes.get_llm", lambda *a, **kw: mock_llm)

    query = "I study 10 h/week, attendance 85%, sleep 7 h. What if I study 15 h/week?"
    state = run_agent(query)

    ml_calls = [r for r in state["tool_results"] if r["tool"] == "predict_performance"]
    assert len(ml_calls) >= 2
    assert state["iteration"] <= config.MAX_ITERATIONS


def test_missing_info_asks_user(monkeypatch):
    """Scenario 4: 'Predict my score' without features -> agent asks for info without crashing."""
    ask_msg = AIMessage(
        content="To predict your performance, please share your key study metrics: How many hours do you study per week and what is your attendance percentage?"
    )
    mock_llm = ScriptedChatModel(scripted_responses=[ask_msg])
    monkeypatch.setattr("src.agent.nodes.get_llm", lambda *a, **kw: mock_llm)

    state = run_agent("Predict my exam score.")
    assert state["iteration"] == 0
    assert len(state["tool_results"]) == 0
    assert "?" in state["messages"][-1].content


def test_tool_error_does_not_crash(monkeypatch):
    """Scenario 5: Forced tool error -> agent handles error dict gracefully and provides an answer."""
    set_mock_force_error(True, "Simulated database connection failure")

    msg1 = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_graph",
                "args": {"query_type": "prerequisites", "params": {"course": "Calculus II"}},
                "id": "call_graph_err",
            }
        ],
    )
    msg2 = AIMessage(
        content="I encountered a temporary database error while retrieving prerequisites. However, typically Calculus I is required before Calculus II."
    )

    mock_llm = ScriptedChatModel(scripted_responses=[msg1, msg2])
    monkeypatch.setattr("src.agent.nodes.get_llm", lambda *a, **kw: mock_llm)

    state = run_agent("What are the prerequisites for Calculus II?")
    assert state["iteration"] <= config.MAX_ITERATIONS
    assert len(state["tool_results"]) == 1
    assert state["tool_results"][0]["output"]["status"] == "error"
    assert "error" in state["tool_results"][0]["output"]


def test_invalid_input_handled(monkeypatch):
    """Scenario 6: Invalid values (e.g. negative study hours) -> validation error returned and handled."""
    msg1 = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "predict_performance",
                "args": {"features": {"Hours_Studied": -5, "Attendance": 150}},
                "id": "call_invalid",
            }
        ],
    )
    msg2 = AIMessage(
        content="The values provided are out of range: study hours must be positive (0-100) and attendance must be between 0 and 100%. Please provide valid numbers."
    )

    mock_llm = ScriptedChatModel(scripted_responses=[msg1, msg2])
    monkeypatch.setattr("src.agent.nodes.get_llm", lambda *a, **kw: mock_llm)

    state = run_agent("I study -5 hours and my attendance is 150%. Predict my score.")
    print("DEBUG test_invalid_input_handled tool_results:", state.get("tool_results"))
    err = state["tool_results"][0]["output"].get("error", "")
    assert "out of allowed range" in err or "must be between" in err or "error" in state["tool_results"][0]["output"]["status"]


def test_iteration_exhaustion_gives_final_answer(monkeypatch):
    """Scenario 7: When agent reaches MAX_ITERATIONS (5), forced final answer is produced."""
    # 5 consecutive tool calls
    tool_msgs = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_documents",
                    "args": {"query": f"question step {i}", "k": 1},
                    "id": f"call_{i}",
                }
            ],
        )
        for i in range(1, 6)
    ]
    forced_final = AIMessage(
        content="Summary after reaching maximum iterations: What was found: preliminary guides. Unresolved: specific details need follow-up."
    )

    mock_llm = ScriptedChatModel(scripted_responses=[*tool_msgs, forced_final])
    monkeypatch.setattr("src.agent.nodes.get_llm", lambda *a, **kw: mock_llm)

    state = run_agent("Help me with an extremely complex multi-part query.")
    assert state["iteration"] == config.MAX_ITERATIONS
    assert "What was found" in state["messages"][-1].content


def test_iteration_never_exceeds_max(monkeypatch):
    """Loop never exceeds MAX_ITERATIONS (5) under any circumstances."""
    infinite_calls = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_documents",
                    "args": {"query": "loop", "k": 1},
                    "id": f"call_loop_{i}",
                }
            ],
        )
        for i in range(10)
    ]
    forced_final = AIMessage(content="Final summary after iteration limit.")

    mock_llm = ScriptedChatModel(scripted_responses=[*infinite_calls, forced_final])
    monkeypatch.setattr("src.agent.nodes.get_llm", lambda *a, **kw: mock_llm)

    state = run_agent("Infinite loop test")
    assert state["iteration"] <= config.MAX_ITERATIONS


def test_final_answer_grounded_in_tool_results(monkeypatch):
    """Scenario 9: Grounding: values and citations in final answer match tool outputs."""
    msg1 = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "predict_performance",
                "args": {"features": {"Hours_Studied": 15, "Attendance": 80}},
                "id": "call_pred",
            }
        ],
    )
    msg2 = AIMessage(
        content="Based on your 15 hours of study and 80% attendance, your predicted score is 65.5 points out of 100."
    )

    mock_llm = ScriptedChatModel(scripted_responses=[msg1, msg2])
    monkeypatch.setattr("src.agent.nodes.get_llm", lambda *a, **kw: mock_llm)

    state = run_agent("Predict my score with 15 hours study and 80% attendance.")
    pred_val = state["tool_results"][0]["output"]["data"]["prediction"]
    assert str(pred_val) in state["messages"][-1].content


if __name__ == "__main__":
    print("Running Agent E2E Tests...")
    config.USE_MOCK_TOOLS = True
    setup_logger("test_agent.log")
    test_funcs = [
        test_early_stop,
        test_what_if_calls_ml_tool_multiple_times,
        test_missing_info_asks_user,
        test_tool_error_does_not_crash,
        test_invalid_input_handled,
        test_iteration_exhaustion_gives_final_answer,
        test_iteration_never_exceeds_max,
        test_final_answer_grounded_in_tool_results,
    ]

    class MonkeyPatcher:
        def setattr(self, target, name_or_val, val=None):
            if val is None:
                parts = target.rsplit(".", 1)
                import importlib
                mod = importlib.import_module(parts[0])
                setattr(mod, parts[1], name_or_val)
            else:
                setattr(target, name_or_val, val)

    mp = MonkeyPatcher()
    passed = 0
    for fn in test_funcs:
        try:
            fn(mp)
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {fn.__name__}: {exc}")
            raise
    print(f"\nSummary: {passed}/{len(test_funcs)} agent tests passed successfully!")
