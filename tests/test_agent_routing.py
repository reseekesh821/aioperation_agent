from langchain_core.messages import AIMessage

import backend.app.agents.agent_graph as agent_graph_module
from backend.app.agents.agent_graph import agent_graph
from backend.app.agents.response_generation import get_last_generation

import pytest
import httpx
from langchain_core.messages import ToolMessage
from backend.app.agents.agent_graph import _final_answer_from_tool


class _NaturalFromFactsLLM:
    """
    Deterministic LLM for graph tests.

    Builds a natural sentence from the verified-facts line so generation
    succeeds without calling Ollama.
    """

    def invoke(self, messages):
        text = ""
        for message in messages:
            content = getattr(message, "content", "")
            if isinstance(content, str) and "Verified facts:" in content:
                text = content.split("Verified facts:", 1)[1].strip()
                break

        if not text:
            return AIMessage(content="unexpected")

        # Keep anchors intact while making the reply look generated.
        return AIMessage(content=f"Here's what I found: {text}")

    def with_structured_output(self, schema, method="json_schema"):
        raise AssertionError("structured output not expected in these tests")


def test_inventory_monitors_via_graph(monkeypatch):
    monkeypatch.setattr(agent_graph_module, "llm", _NaturalFromFactsLLM())

    result = agent_graph.invoke({
        "messages": [("user", "How many monitors are available?")]
    })

    tools_used = [m.name for m in result["messages"] if m.type == "tool"]
    answer = result["messages"][-1].content
    meta = get_last_generation()

    assert tools_used == ["inventory_lookup"]
    assert "5" in answer
    assert "Monitor" in answer
    assert answer != "Monitor: 5 units available."
    assert meta is not None
    assert meta.used_fallback is False


def test_inventory_mice_zero_stock(monkeypatch):
    monkeypatch.setattr(agent_graph_module, "llm", _NaturalFromFactsLLM())

    result = agent_graph.invoke({
        "messages": [("user", "Do you have any mice?")]
    })

    answer = result["messages"][-1].content
    tools_used = [m.name for m in result["messages"] if m.type == "tool"]

    assert tools_used == ["inventory_lookup"]
    assert "0" in answer
    assert "out of stock" in answer.lower()
    assert get_last_generation().used_fallback is False


def test_unknown_product_via_graph():
    result = agent_graph.invoke({
        "messages": [("user", "How many units are available?")]
    })

    answer = result["messages"][-1].content
    tools_used = [m.name for m in result["messages"] if m.type == "tool"]

    assert tools_used == []
    assert "product" in answer.lower()


@pytest.mark.parametrize(
    "question",
    [
        "What is the status of order 1003?",
        "What is the order status of 1003?",
        "What's happening with order 1003?",
    ],
)
def test_order_status_via_graph(question, monkeypatch):
    monkeypatch.setattr(agent_graph_module, "llm", _NaturalFromFactsLLM())

    result = agent_graph.invoke({
        "messages": [("user", question)]
    })

    tools_used = [m.name for m in result["messages"] if m.type == "tool"]
    answer = result["messages"][-1].content

    assert tools_used == ["order_lookup"]
    assert "Delivered" in answer
    assert "1003" in answer
    assert get_last_generation().used_fallback is False


def test_check_order_phrasing_via_graph(monkeypatch):
    monkeypatch.setattr(agent_graph_module, "llm", _NaturalFromFactsLLM())

    result = agent_graph.invoke({
        "messages": [("user", "Check order 1001.")]
    })

    tools_used = [m.name for m in result["messages"] if m.type == "tool"]
    answer = result["messages"][-1].content

    assert tools_used == ["order_lookup"]
    assert "1001" in answer
    assert "Shipped" in answer


def test_nonexistent_order_via_graph(monkeypatch):
    monkeypatch.setattr(agent_graph_module, "llm", _NaturalFromFactsLLM())

    result = agent_graph.invoke({
        "messages": [("user", "What is the order status of 9999?")]
    })

    tools_used = [m.name for m in result["messages"] if m.type == "tool"]
    answer = result["messages"][-1].content

    assert tools_used == ["order_lookup"]
    assert "not found" in answer.lower()


def test_refund_eligibility_via_graph(monkeypatch):
    monkeypatch.setattr(agent_graph_module, "llm", _NaturalFromFactsLLM())

    result = agent_graph.invoke({
        "messages": [("user", "Is order 1001 eligible for a refund?")]
    })

    tools_used = [m.name for m in result["messages"] if m.type == "tool"]
    answer = result["messages"][-1].content

    assert tools_used == ["refund_lookup"]
    assert "unavailable" in answer.lower() or "not" in answer.lower()
    assert get_last_generation().used_fallback is False


def test_refund_for_order_phrasing_stays_on_refund_tool(monkeypatch):
    monkeypatch.setattr(agent_graph_module, "llm", _NaturalFromFactsLLM())

    result = agent_graph.invoke({
        "messages": [("user", "Can I get a refund for order 1001?")]
    })

    tools_used = [m.name for m in result["messages"] if m.type == "tool"]
    assert tools_used == ["refund_lookup"]


def test_unclear_request_asks_for_clarification():
    result = agent_graph.invoke({
        "messages": [("user", "Hello there")]
    })

    tools_used = [m.name for m in result["messages"] if m.type == "tool"]
    answer = result["messages"][-1].content

    assert tools_used == []
    assert "inventory" in answer.lower() or "order" in answer.lower()


def test_fallback_when_generation_fails_validation(monkeypatch):
    class BadLLM:
        def invoke(self, messages):
            return AIMessage(
                content="Unlimited stock plus lifetime warranty and store credit."
            )

    monkeypatch.setattr(agent_graph_module, "llm", BadLLM())

    result = agent_graph.invoke({
        "messages": [("user", "How many monitors are available?")]
    })

    answer = result["messages"][-1].content
    meta = get_last_generation()

    assert answer == "Monitor: 5 units available."
    assert meta.used_fallback is True
    assert meta.attempts == 2

def test_policy_connection_failure_propagates(monkeypatch):
    def failing_policy_plan(question, evidence, llm):
        raise httpx.ConnectError("Ollama is offline")

    monkeypatch.setattr(
        "backend.app.agents.agent_graph.select_policy_evidence",
        lambda question, chunks: ["Verified refund policy"]
    )

    monkeypatch.setattr(
        "backend.app.agents.agent_graph.generate_policy_plan",
        failing_policy_plan
    )

    tool_message = ToolMessage(
        content="Refund requests must be made within 30 days.",
        name="policy_lookup",
        tool_call_id="test-call"
    )

    with pytest.raises(httpx.ConnectError):
        _final_answer_from_tool(
            "What is the refund deadline?",
            tool_message
        )
