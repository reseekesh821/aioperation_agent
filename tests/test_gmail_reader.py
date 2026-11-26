import base64
from pathlib import Path

import pytest

from backend.app.integrations import gmail_reader as gmail_reader_module
from backend.app.integrations.gmail_reader import (
    GmailAuthError,
    GmailLabelNotFoundError,
    GmailMessageNotAllowedError,
    extract_plain_text_body,
    fetch_support_email_by_id,
    fetch_support_emails,
    html_to_plain_text,
    load_gmail_credentials,
)


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def test_html_to_plain_text_strips_markup_and_scripts():
    html = (
        "<html><body><p>Hello <b>customer</b></p>"
        "<script>alert('x')</script>"
        "<p>Order 1003</p></body></html>"
    )
    text = html_to_plain_text(html)
    assert "Hello customer" in text
    assert "Order 1003" in text
    assert "alert" not in text
    assert "<" not in text


def test_extract_plain_text_prefers_text_plain_over_html():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {
                "mimeType": "text/plain",
                "body": {"data": _b64("Plain body line")},
            },
            {
                "mimeType": "text/html",
                "body": {"data": _b64("<p>HTML body line</p>")},
            },
        ],
    }
    assert extract_plain_text_body(payload) == "Plain body line"


def test_extract_plain_text_falls_back_to_html_and_skips_attachments():
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "mimeType": "text/html",
                "body": {"data": _b64("<p>Only HTML body</p>")},
            },
            {
                "filename": "invoice.pdf",
                "mimeType": "application/pdf",
                "body": {"attachmentId": "att-1", "data": _b64("not-used")},
            },
        ],
    }
    assert extract_plain_text_body(payload) == "Only HTML body"


def test_extract_nested_multipart_plain_text():
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": _b64("Nested plain body")},
                    },
                    {
                        "mimeType": "text/html",
                        "body": {"data": _b64("<p>Nested HTML</p>")},
                    },
                ],
            }
        ],
    }
    assert extract_plain_text_body(payload) == "Nested plain body"


def test_load_gmail_credentials_missing_token(tmp_path: Path):
    with pytest.raises(GmailAuthError, match="authorization is missing"):
        load_gmail_credentials(token_path=tmp_path / "missing-token.json")


def test_load_gmail_credentials_refreshes_expired_token(monkeypatch, tmp_path: Path):
    token_path = tmp_path / "token.json"
    token_path.write_text("{}")

    class FakeCreds:
        valid = False
        expired = True
        refresh_token = "refresh-token"

        def refresh(self, request):  # noqa: ANN001
            self.valid = True

        def to_json(self):
            return '{"token":"refreshed"}'

    monkeypatch.setattr(
        gmail_reader_module.Credentials,
        "from_authorized_user_file",
        lambda path, scopes: FakeCreds(),
    )
    monkeypatch.setattr(gmail_reader_module, "Request", lambda: object())

    creds = load_gmail_credentials(token_path=token_path)
    assert creds.valid is True
    assert token_path.read_text() == '{"token":"refreshed"}'


class _FakeExecute:
    def __init__(self, payload):
        self._payload = payload

    def execute(self):
        return self._payload


class _FakeMessages:
    def __init__(self, listed, details_by_id):
        self._listed = listed
        self._details_by_id = details_by_id

    def list(self, userId, labelIds, maxResults):  # noqa: N803, ANN001
        assert userId == "me"
        assert maxResults == 5
        assert labelIds == ["Label_123"]
        return _FakeExecute(self._listed)

    def get(self, userId, id, format):  # noqa: A002, N803, ANN001
        assert userId == "me"
        assert format == "full"
        return _FakeExecute(self._details_by_id[id])


class _FakeLabels:
    def list(self, userId):  # noqa: N803, ANN001
        assert userId == "me"
        return _FakeExecute({
            "labels": [
                {"id": "INBOX", "name": "INBOX"},
                {"id": "Label_123", "name": "AI-Operations-Test"},
            ]
        })


class _FakeUsers:
    def __init__(self, listed, details_by_id):
        self._messages = _FakeMessages(listed, details_by_id)
        self._labels = _FakeLabels()

    def labels(self):
        return self._labels

    def messages(self):
        return self._messages


class _FakeGmailService:
    def __init__(self, listed, details_by_id):
        self._users = _FakeUsers(listed, details_by_id)

    def users(self):
        return self._users


def test_fetch_support_emails_returns_structured_messages():
    details = {
        "msg-1": {
            "id": "msg-1",
            "payload": {
                "headers": [
                    {"name": "From", "value": "customer@example.com"},
                    {"name": "Subject", "value": "Where is my order?"},
                ],
                "mimeType": "text/plain",
                "body": {"data": _b64("Please check order 1003.")},
            },
        }
    }
    service = _FakeGmailService(
        listed={"messages": [{"id": "msg-1"}]},
        details_by_id=details,
    )

    emails = fetch_support_emails(service=service)
    assert emails[0]["id"] == "msg-1"
    assert emails[0]["sender"] == "customer@example.com"
    assert emails[0]["subject"] == "Where is my order?"
    assert emails[0]["body"] == "Please check order 1003."
    assert emails[0]["thread_id"] == ""
    assert "rfc822_message_id" in emails[0]


def test_fetch_support_emails_label_missing():
    class MissingLabelService:
        def users(self):
            return self

        def labels(self):
            return self

        def list(self, userId):  # noqa: N803, ANN001
            return _FakeExecute({"labels": [{"id": "INBOX", "name": "INBOX"}]})

    with pytest.raises(GmailLabelNotFoundError, match="AI-Operations-Test"):
        fetch_support_emails(service=MissingLabelService())


def test_fetch_support_email_by_id_requires_permitted_label():
    details = {
        "msg-ok": {
            "id": "msg-ok",
            "labelIds": ["Label_123"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "a@example.com"},
                    {"name": "Subject", "value": "Help"},
                ],
                "mimeType": "text/plain",
                "body": {"data": _b64("Body text")},
            },
        },
        "msg-bad": {
            "id": "msg-bad",
            "labelIds": ["INBOX"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "a@example.com"},
                    {"name": "Subject", "value": "Help"},
                ],
                "mimeType": "text/plain",
                "body": {"data": _b64("Body text")},
            },
        },
    }

    class ByIdService:
        def __init__(self):
            self._labels = _FakeLabels()
            self._details = details

        def users(self):
            return self

        def labels(self):
            return self._labels

        def messages(self):
            return self

        def get(self, userId, id, format):  # noqa: A002, N803, ANN001
            return _FakeExecute(self._details[id])

    service = ByIdService()
    parsed = fetch_support_email_by_id("msg-ok", service=service)
    assert parsed["id"] == "msg-ok"
    assert parsed["body"] == "Body text"

    with pytest.raises(GmailMessageNotAllowedError, match="permitted"):
        fetch_support_email_by_id("msg-bad", service=service)
