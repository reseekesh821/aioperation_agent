import base64
from email import message_from_bytes

import pytest
from fastapi.testclient import TestClient

from backend.app.integrations.gmail_drafts import (
    GmailDraftValidationError,
    build_reply_mime,
    build_reply_subject,
    derive_reply_recipient,
    extract_email_address,
    save_support_reply_draft,
    validate_draft_text,
)
from backend.app.integrations.gmail_reader import (
    GmailAuthError,
    GmailApiError,
    GmailMessageNotAllowedError,
)
from backend.app.main import app

client = TestClient(app)


def test_validate_draft_text_rejects_empty_and_oversized():
    with pytest.raises(GmailDraftValidationError, match="empty"):
        validate_draft_text("   ")
    with pytest.raises(GmailDraftValidationError, match="exceeds"):
        validate_draft_text("x" * 10001)
    assert validate_draft_text("  Hello  ") == "Hello"


def test_extract_and_derive_recipient():
    assert extract_email_address("Ada <customer@example.com>") == "customer@example.com"
    email = {
        "reply_to": "Support Desk <ops@example.com>",
        "sender": "Customer <customer@example.com>",
    }
    assert derive_reply_recipient(email) == "ops@example.com"
    assert build_reply_subject("Refund help") == "Re: Refund help"
    assert build_reply_subject("Re: Refund help") == "Re: Refund help"


def test_build_reply_mime_preserves_threading_headers():
    mime = build_reply_mime(
        draft_text="Edited operator reply",
        recipient="customer@example.com",
        subject="Order question",
        rfc822_message_id="<abc@mail.gmail.com>",
        references="<root@mail.gmail.com>",
    )
    assert mime["To"] == "customer@example.com"
    assert mime["Subject"] == "Re: Order question"
    assert mime["In-Reply-To"] == "<abc@mail.gmail.com>"
    assert "<root@mail.gmail.com>" in mime["References"]
    assert "<abc@mail.gmail.com>" in mime["References"]
    assert mime.get_payload(decode=True).decode("utf-8") == "Edited operator reply"


class _FakeExecute:
    def __init__(self, payload):
        self._payload = payload

    def execute(self):
        return self._payload


class _FakeDraftsApi:
    def __init__(self):
        self.created = []
        self.send_called = False

    def create(self, userId, body):  # noqa: N803, ANN001
        assert userId == "me"
        self.created.append(body)
        return _FakeExecute({
            "id": "draft-123",
            "message": {
                "id": "msg-draft-123",
                "threadId": body["message"].get("threadId") or "thread-fallback",
            },
        })

    def send(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.send_called = True
        raise AssertionError("drafts.send must never be called")


class _FakeMessagesApi:
    def send(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("messages.send must never be called")


class _FakeUsers:
    def __init__(self, drafts_api):
        self._drafts = drafts_api
        self._messages = _FakeMessagesApi()

    def drafts(self):
        return self._drafts

    def messages(self):
        return self._messages


class _FakeGmailService:
    def __init__(self):
        self.drafts_api = _FakeDraftsApi()
        self._users = _FakeUsers(self.drafts_api)

    def users(self):
        return self._users


def _original_email():
    return {
        "id": "msg-1",
        "thread_id": "thread-9",
        "sender": "Customer <customer@example.com>",
        "reply_to": "",
        "subject": "Refund for order 1003",
        "body": "Please refund order 1003.",
        "rfc822_message_id": "<orig@mail.gmail.com>",
        "references": "",
    }


def test_save_support_reply_draft_uses_edited_text_and_thread():
    service = _FakeGmailService()
    edited = "Human-edited reply for order 1003. Please verify ownership."

    result = save_support_reply_draft(
        "msg-1",
        edited,
        fetch_email=lambda message_id, service=None: _original_email(),
        service=service,
    )

    assert result["draft_id"] == "draft-123"
    assert result["thread_id"] == "thread-9"
    assert result["recipient"] == "customer@example.com"
    assert result["subject"] == "Re: Refund for order 1003"
    assert result["status"] == "saved_not_sent"
    assert result["sent"] is False
    assert service.drafts_api.send_called is False
    assert len(service.drafts_api.created) == 1

    raw = service.drafts_api.created[0]["message"]["raw"]
    parsed = message_from_bytes(base64.urlsafe_b64decode(raw.encode("ascii")))
    body = parsed.get_payload(decode=True).decode("utf-8")
    assert body == edited
    assert "Please refund order 1003." not in body
    assert parsed["To"] == "customer@example.com"
    assert parsed["In-Reply-To"] == "<orig@mail.gmail.com>"
    assert service.drafts_api.created[0]["message"]["threadId"] == "thread-9"


def test_save_support_reply_draft_rejects_unlabeled_message():
    def fetch_unlabeled(message_id, service=None):
        raise GmailMessageNotAllowedError(
            "Gmail message is not in the permitted 'AI-Operations-Test' label."
        )

    with pytest.raises(GmailMessageNotAllowedError):
        save_support_reply_draft(
            "msg-bad",
            "Valid edited text",
            fetch_email=fetch_unlabeled,
            service=_FakeGmailService(),
        )


def test_save_support_reply_draft_auth_error_on_forbidden_create():
    from googleapiclient.errors import HttpError
    import httplib2

    class ForbiddenCreate:
        def create(self, userId, body):  # noqa: N803, ANN001
            raise HttpError(
                resp=httplib2.Response({"status": "403"}),
                content=b"insufficient permissions",
            )

        def send(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("send must never be called")

    class Service:
        def users(self):
            return self

        def drafts(self):
            return ForbiddenCreate()

        def messages(self):
            return _FakeMessagesApi()

    with pytest.raises(GmailAuthError, match="compose|expired|authorization"):
        save_support_reply_draft(
            "msg-1",
            "Valid edited text",
            fetch_email=lambda message_id, service=None: _original_email(),
            service=Service(),
        )


def test_save_draft_endpoint_success(monkeypatch):
    monkeypatch.setattr(
        "backend.app.main.save_support_reply_draft",
        lambda message_id, draft_text: {
            "message_id": message_id,
            "draft_id": "draft-xyz",
            "thread_id": "thread-1",
            "recipient": "customer@example.com",
            "subject": "Re: Help",
            "status": "saved_not_sent",
            "sent": False,
            "saved_text": draft_text,
        },
    )

    edited = "Operator edited this reply carefully."
    response = client.post(
        "/support/emails/msg-1/save-draft",
        json={"draft_text": edited},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["draft_id"] == "draft-xyz"
    assert payload["sent"] is False
    assert payload["status"] == "saved_not_sent"
    assert "token" not in response.text.casefold()


def test_save_draft_endpoint_passes_edited_text(monkeypatch):
    captured = {}

    def fake_save(message_id, draft_text):
        captured["message_id"] = message_id
        captured["draft_text"] = draft_text
        return {
            "message_id": message_id,
            "draft_id": "draft-1",
            "thread_id": "thread-1",
            "recipient": "customer@example.com",
            "subject": "Re: Help",
            "status": "saved_not_sent",
            "sent": False,
        }

    monkeypatch.setattr("backend.app.main.save_support_reply_draft", fake_save)
    edited = "This is the human-edited body, not the AI draft."
    response = client.post(
        "/support/emails/msg-22/save-draft",
        json={"draft_text": edited},
    )
    assert response.status_code == 200
    assert captured == {"message_id": "msg-22", "draft_text": edited}


def test_save_draft_endpoint_invalid_body():
    response = client.post(
        "/support/emails/msg-1/save-draft",
        json={"draft_text": "   "},
    )
    assert response.status_code == 422


def test_save_draft_endpoint_unlabeled_message(monkeypatch):
    def raise_not_allowed(message_id, draft_text):
        raise GmailMessageNotAllowedError(
            "Gmail message is not in the permitted 'AI-Operations-Test' label."
        )

    monkeypatch.setattr("backend.app.main.save_support_reply_draft", raise_not_allowed)
    response = client.post(
        "/support/emails/msg-1/save-draft",
        json={"draft_text": "Edited reply"},
    )
    assert response.status_code == 403


def test_save_draft_endpoint_auth_and_api_errors(monkeypatch):
    def raise_auth(message_id, draft_text):
        raise GmailAuthError("Gmail compose permission is missing.")

    monkeypatch.setattr("backend.app.main.save_support_reply_draft", raise_auth)
    response = client.post(
        "/support/emails/msg-1/save-draft",
        json={"draft_text": "Edited reply"},
    )
    assert response.status_code == 401

    def raise_api(message_id, draft_text):
        raise GmailApiError("Failed to create a Gmail draft.")

    monkeypatch.setattr("backend.app.main.save_support_reply_draft", raise_api)
    response = client.post(
        "/support/emails/msg-1/save-draft",
        json={"draft_text": "Edited reply"},
    )
    assert response.status_code == 502
