from fastapi.testclient import TestClient

from backend.app.integrations.gmail_reader import (
    GmailApiError,
    GmailAuthError,
    GmailLabelNotFoundError,
)
from backend.app.main import app

client = TestClient(app)


def test_support_emails_endpoint_success(monkeypatch):
    monkeypatch.setattr(
        "backend.app.main.fetch_support_emails",
        lambda: [{
            "id": "msg-1",
            "sender": "customer@example.com",
            "subject": "Refund question",
            "body": "Can I get a refund for order 1003?",
        }],
    )

    response = client.get("/support/emails")
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "emails": [{
            "id": "msg-1",
            "sender": "customer@example.com",
            "subject": "Refund question",
            "body": "Can I get a refund for order 1003?",
        }]
    }
    # Never expose credential material.
    assert "token" not in response.text.casefold()
    assert "refresh_token" not in response.text.casefold()
    assert "credentials" not in response.text.casefold()


def test_support_emails_endpoint_auth_error(monkeypatch):
    def raise_auth():
        raise GmailAuthError("Gmail authorization is missing.")

    monkeypatch.setattr("backend.app.main.fetch_support_emails", raise_auth)

    response = client.get("/support/emails")
    assert response.status_code == 401
    assert "authorization is missing" in response.json()["detail"].lower()


def test_support_emails_endpoint_label_missing(monkeypatch):
    def raise_label():
        raise GmailLabelNotFoundError(
            "Gmail label 'AI-Operations-Test' was not found."
        )

    monkeypatch.setattr("backend.app.main.fetch_support_emails", raise_label)

    response = client.get("/support/emails")
    assert response.status_code == 404
    assert "AI-Operations-Test" in response.json()["detail"]


def test_support_emails_endpoint_api_error(monkeypatch):
    def raise_api():
        raise GmailApiError("Failed to list labeled Gmail messages.")

    monkeypatch.setattr("backend.app.main.fetch_support_emails", raise_api)

    response = client.get("/support/emails")
    assert response.status_code == 502
    assert "Failed to list" in response.json()["detail"]
