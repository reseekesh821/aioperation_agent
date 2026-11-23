import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend.app.main import app
import backend.app.agents.agent_graph as agent_graph_module
from backend.app.agents.response_generation import get_last_generation

client = TestClient(app)


class _NaturalFromFactsLLM:
    def invoke(self, messages):
        text = ""
        for message in messages:
            content = getattr(message, "content", "")
            if isinstance(content, str) and "Verified facts:" in content:
                text = content.split("Verified facts:", 1)[1].strip()
                break
        return AIMessage(content=f"Sure — {text}")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_inventory_endpoint_plural():
    response = client.get("/inventory/monitors")
    assert response.status_code == 200
    assert response.json() == {"product": "Monitor", "quantity": 5}


def test_inventory_endpoint_zero_stock():
    response = client.get("/inventory/mice")
    assert response.status_code == 200
    assert response.json() == {"product": "Mouse", "quantity": 0}


def test_inventory_endpoint_unknown():
    response = client.get("/inventory/laptop")
    assert response.status_code == 404


def test_order_endpoint():
    response = client.get("/orders/1003")
    assert response.status_code == 200
    assert response.json()["status"] == "Delivered"


def test_chat_endpoint_mocked_graph(monkeypatch):
    class FakeResult:
        def __getitem__(self, key):
            assert key == "messages"

            class Msg:
                content = "There are 5 monitors available."

            return [Msg()]

    monkeypatch.setattr(
        "backend.app.main.agent_graph.invoke",
        lambda state: FakeResult(),
    )

    response = client.post("/chat", json={"message": "How many monitors?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "There are 5 monitors available."


def test_chat_endpoint_generates_from_verified_facts(monkeypatch):
    monkeypatch.setattr(agent_graph_module, "llm", _NaturalFromFactsLLM())

    response = client.post(
        "/chat",
        json={"message": "How many monitors are available?"},
    )
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "5" in answer
    assert "Monitor" in answer
    assert answer != "Monitor: 5 units available."
    assert get_last_generation().used_fallback is False

def test_chat_ollama_unavailable(monkeypatch):
    class FailingGraph:
        def invoke(self, state):
            raise ConnectionError("Ollama is offline")

    monkeypatch.setattr(
        "backend.app.main.agent_graph",
        FailingGraph()
    )

    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"message": "How many monitors are available?"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "AI service is temporarily unavailable."
    )

def test_chat_httpx_connection_error(monkeypatch):
    class FailingGraph:
        def invoke(self, state):
            raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(
        "backend.app.main.agent_graph",
        FailingGraph()
    )

    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"message": "Check inventory for Monitor"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "AI service is temporarily unavailable."
    )

def test_chat_ollama_timeout(monkeypatch):
    class SlowGraph:
        def invoke(self, state):
            raise httpx.ReadTimeout("Ollama response timed out")

    monkeypatch.setattr(
        "backend.app.main.agent_graph",
        SlowGraph()
    )

    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"message": "What is the refund deadline?"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "AI service is temporarily unavailable."
    )


def test_chat_wrapped_connection_error_returns_503(monkeypatch):
    class FailingGraph:
        def invoke(self, state):
            err = RuntimeError("chat invoke failed")
            err.__cause__ = ConnectionError("Ollama is offline")
            raise err

    monkeypatch.setattr("backend.app.main.agent_graph", FailingGraph())

    response = client.post(
        "/chat",
        json={"message": "How many monitors are available?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "AI service is temporarily unavailable."
    )
    assert "Ollama" not in response.json()["detail"]
    assert "offline" not in response.json()["detail"]


def test_chat_non_ai_errors_are_not_mapped_to_503(monkeypatch):
    class FailingGraph:
        def invoke(self, state):
            raise ValueError("unexpected bug")

    monkeypatch.setattr("backend.app.main.agent_graph", FailingGraph())

    with pytest.raises(ValueError, match="unexpected bug"):
        client.post(
            "/chat",
            json={"message": "How many monitors are available?"},
        )
