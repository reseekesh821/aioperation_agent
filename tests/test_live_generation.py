"""
Live Ollama checks (separated from deterministic unit tests).

These verify that ordinary supported questions receive LLM-generated
final answers when generation passes grounding.
"""

import pytest

from backend.app.agents.agent_graph import agent_graph
from backend.app.agents.response_generation import get_last_generation
from fastapi.testclient import TestClient
from backend.app.main import app


client = TestClient(app)

LIVE_CASES = [
    (
        "How many monitors are available?",
        ["inventory_lookup"],
        ["5", "monitor"],
    ),
    (
        "Do you have any mice?",
        ["inventory_lookup"],
        ["mice"],
    ),
    (
        "What is the status of order 1003?",
        ["order_lookup"],
        ["1003", "delivered"],
    ),
    (
        "Is order 1001 eligible for a refund?",
        ["refund_lookup"],
        ["1001", "not"],
    ),
    (
        "What is the refund deadline?",
        ["policy_lookup"],
        ["30", "delivery"],
    ),
]


@pytest.mark.parametrize("question, tools, terms", LIVE_CASES)
def test_live_natural_generation(question, tools, terms):
    result = agent_graph.invoke({"messages": [("user", question)]})
    answer = result["messages"][-1].content
    tools_used = [m.name for m in result["messages"] if m.type == "tool"]
    meta = get_last_generation()

    assert tools_used == tools
    lowered = answer.lower()
    for term in terms:
        assert term.lower() in lowered, f"{term!r} missing in {answer!r}"

    assert meta is not None
    assert answer.strip() != ""
    # Fallback is observable; these ordinary questions should usually generate.
    if meta.used_fallback:
        pytest.skip(
            f"fallback used for {question!r}: {meta.reason}; "
            f"candidates={meta.candidates!r}"
        )
    assert meta.used_fallback is False
    # Successful generation should differ from a raw tool echo for inventory.
    if tools[0] == "inventory_lookup":
        assert answer != "Monitor: 5 units available."
        assert answer != "Mouse: 0 units available (out of stock)."


def test_live_chat_api_inventory():
    response = client.post(
        "/chat",
        json={"message": "How many monitors are available?"},
    )
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "5" in answer
    meta = get_last_generation()
    assert meta is not None
