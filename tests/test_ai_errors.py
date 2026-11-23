import httpx
import pytest

from backend.app.agents.ai_errors import is_ai_service_error
from backend.app.agents.response_generation import generate_grounded_response
from backend.app.agents.agent_graph import _final_answer_from_tool, POLICY_FALLBACK
from langchain_core.messages import ToolMessage


def test_detects_connection_error():
    assert is_ai_service_error(ConnectionError("offline"))


def test_detects_httpx_timeout():
    assert is_ai_service_error(httpx.ReadTimeout("timed out"))


def test_detects_wrapped_connect_error():
    cause = httpx.ConnectError("refused")
    wrapped = ValueError("Failed to connect to Ollama")
    wrapped.__cause__ = cause
    assert is_ai_service_error(wrapped)


def test_ignores_ordinary_value_error():
    assert not is_ai_service_error(ValueError("LLM selected unsupported policy IDs"))


def test_generate_propagates_connection_error():
    class FailingLLM:
        def invoke(self, messages):
            raise ConnectionError("Ollama is offline")

    with pytest.raises(ConnectionError):
        generate_grounded_response(
            "How many monitors are available?",
            "Monitor: 5 units available.",
            FailingLLM(),
        )


def test_generate_propagates_wrapped_connect_error():
    class FailingLLM:
        def invoke(self, messages):
            err = ValueError("Failed to connect to Ollama")
            err.__cause__ = httpx.ConnectError("refused")
            raise err

    with pytest.raises(ValueError):
        generate_grounded_response(
            "How many monitors are available?",
            "Monitor: 5 units available.",
            FailingLLM(),
        )


def test_policy_planning_falls_back_on_non_connectivity_errors(monkeypatch):
    monkeypatch.setattr(
        "backend.app.agents.agent_graph.select_policy_evidence",
        lambda question, chunks: [{"id": "refund_1", "text": "deadline"}],
    )

    def boom(question, evidence, llm):
        raise ValueError("LLM selected unsupported policy IDs")

    monkeypatch.setattr(
        "backend.app.agents.agent_graph.generate_policy_plan",
        boom,
    )

    tool_message = ToolMessage(
        content="1. Standard refund requests can be submitted only after delivery.",
        name="policy_lookup",
        tool_call_id="test-call",
    )
    answer = _final_answer_from_tool("What is the refund deadline?", tool_message)
    assert answer == POLICY_FALLBACK


def test_policy_planning_propagates_wrapped_ai_errors(monkeypatch):
    monkeypatch.setattr(
        "backend.app.agents.agent_graph.select_policy_evidence",
        lambda question, chunks: [{"id": "refund_1", "text": "deadline"}],
    )

    def boom(question, evidence, llm):
        err = RuntimeError("chat failed")
        err.__cause__ = ConnectionError("Ollama is offline")
        raise err

    monkeypatch.setattr(
        "backend.app.agents.agent_graph.generate_policy_plan",
        boom,
    )

    tool_message = ToolMessage(
        content="1. Standard refund requests can be submitted only after delivery.",
        name="policy_lookup",
        tool_call_id="test-call",
    )
    with pytest.raises(RuntimeError):
        _final_answer_from_tool("What is the refund deadline?", tool_message)
