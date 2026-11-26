"""Read-only Gmail helpers for the local Customer Support Inbox."""

from __future__ import annotations

import base64
import logging
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
TOKEN_FILE = ROOT / "token.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]
LABEL_NAME = "AI-Operations-Test"
MAX_MESSAGES = 5


class GmailAuthError(Exception):
    """Missing, invalid, or unrefreshable Gmail authorization."""


class GmailLabelNotFoundError(Exception):
    """Configured Gmail label is not present in the account."""


class GmailApiError(Exception):
    """Unexpected Gmail API failure."""


class GmailMessageNotAllowedError(Exception):
    """Requested message is missing or outside the permitted support label."""


class _HTMLTextExtractor(HTMLParser):
    """Convert HTML email bodies into plain text without executing markup."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if tag in {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in {"p", "div", "li", "tr"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def _decode_body_data(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    return raw.decode("utf-8", errors="replace")


def html_to_plain_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html or "")
    parser.close()
    text = unescape(parser.text())
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _header_value(headers: list[dict[str, str]], name: str) -> str:
    target = name.casefold()
    for header in headers:
        if header.get("name", "").casefold() == target:
            return (header.get("value") or "").strip()
    return ""


def _collect_text_parts(
    payload: dict[str, Any] | None,
    *,
    plain_parts: list[str],
    html_parts: list[str],
) -> None:
    if not payload:
        return

    filename = (payload.get("filename") or "").strip()
    mime_type = (payload.get("mimeType") or "").casefold()
    body = payload.get("body") or {}
    data = body.get("data")

    # Skip attachment payloads; only walk nested parts for containers.
    if filename:
        for child in payload.get("parts") or []:
            _collect_text_parts(child, plain_parts=plain_parts, html_parts=html_parts)
        return

    if data and mime_type == "text/plain":
        plain_parts.append(_decode_body_data(data))
    elif data and mime_type == "text/html":
        html_parts.append(_decode_body_data(data))
    elif data and not mime_type.startswith("multipart/"):
        # Rare non-multipart bodies without an explicit text/* type.
        plain_parts.append(_decode_body_data(data))

    for child in payload.get("parts") or []:
        _collect_text_parts(child, plain_parts=plain_parts, html_parts=html_parts)


def extract_plain_text_body(payload: dict[str, Any] | None) -> str:
    """Prefer text/plain; fall back to sanitized HTML. Never use snippets."""
    plain_parts: list[str] = []
    html_parts: list[str] = []
    _collect_text_parts(payload, plain_parts=plain_parts, html_parts=html_parts)

    plain = "\n\n".join(part.strip() for part in plain_parts if part.strip()).strip()
    if plain:
        return plain

    html = "\n\n".join(part.strip() for part in html_parts if part.strip()).strip()
    if html:
        return html_to_plain_text(html)

    return ""


def load_gmail_credentials(
    *,
    token_path: Path = TOKEN_FILE,
) -> Credentials:
    """
    Load and refresh stored Gmail credentials.

    Never launches interactive OAuth during API requests.
    """
    if not token_path.exists():
        raise GmailAuthError(
            "Gmail authorization is missing. Run scripts/test_gmail_connection.py "
            "once on localhost to create token.json."
        )

    try:
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    except Exception as exc:  # noqa: BLE001
        raise GmailAuthError(
            "Gmail authorization could not be loaded from token.json."
        ) from exc

    if creds.valid:
        return creds

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            raise GmailAuthError(
                "Gmail authorization expired and could not be refreshed. "
                "Re-run scripts/test_gmail_connection.py on localhost."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise GmailAuthError(
                "Gmail authorization refresh failed."
            ) from exc

        try:
            token_path.write_text(creds.to_json())
            token_path.chmod(0o600)
        except OSError:
            # Refresh succeeded; failing to persist should not block reads.
            logger.warning("Unable to persist refreshed Gmail token to disk.")
        return creds

    raise GmailAuthError(
        "Gmail authorization is invalid. Re-run scripts/test_gmail_connection.py "
        "on localhost."
    )


def _build_gmail_service(creds: Credentials):
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _find_label_id(service, label_name: str = LABEL_NAME) -> str:
    try:
        labels = (
            service.users()
            .labels()
            .list(userId="me")
            .execute()
            .get("labels", [])
        )
    except HttpError as exc:
        raise GmailApiError("Failed to list Gmail labels.") from exc

    for label in labels:
        if label.get("name") == label_name:
            label_id = label.get("id")
            if label_id:
                return label_id

    raise GmailLabelNotFoundError(
        f"Gmail label '{label_name}' was not found in this account."
    )


def _parse_message(details: dict[str, Any]) -> dict[str, str]:
    headers = details.get("payload", {}).get("headers", []) or []
    message_id = details.get("id") or ""
    sender = _header_value(headers, "From") or "(Unknown sender)"
    subject = _header_value(headers, "Subject") or "(No subject)"
    body = extract_plain_text_body(details.get("payload"))
    reply_to = _header_value(headers, "Reply-To")
    rfc822_message_id = (
        _header_value(headers, "Message-ID")
        or _header_value(headers, "Message-Id")
    )
    references = _header_value(headers, "References")

    return {
        "id": message_id,
        "thread_id": details.get("threadId") or "",
        "sender": sender,
        "reply_to": reply_to,
        "subject": subject,
        "body": body,
        "rfc822_message_id": rfc822_message_id,
        "references": references,
    }


def _validate_message_id(message_id: str) -> str:
    cleaned = (message_id or "").strip()
    if not cleaned or not re.fullmatch(r"[A-Za-z0-9_-]+", cleaned):
        raise GmailApiError("Invalid Gmail message ID.")
    return cleaned


def _get_message_details(service, message_id: str) -> dict[str, Any]:
    try:
        return (
            service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
    except HttpError as exc:
        status = getattr(getattr(exc, "resp", None), "status", None)
        if status == 404:
            raise GmailMessageNotAllowedError(
                "Gmail message was not found or is not available."
            ) from exc
        raise GmailApiError("Failed to fetch a labeled Gmail message.") from exc


def fetch_support_email_by_id(
    message_id: str,
    *,
    label_name: str = LABEL_NAME,
    token_path: Path = TOKEN_FILE,
    service=None,
) -> dict[str, str]:
    """
    Fetch one support email by Gmail ID after verifying the permitted label.

    Never trusts browser-supplied email text; always reloads from Gmail.
    """
    message_id = _validate_message_id(message_id)
    gmail = service
    if gmail is None:
        creds = load_gmail_credentials(token_path=token_path)
        gmail = _build_gmail_service(creds)

    label_id = _find_label_id(gmail, label_name=label_name)
    details = _get_message_details(gmail, message_id)
    label_ids = details.get("labelIds") or []
    if label_id not in label_ids:
        raise GmailMessageNotAllowedError(
            f"Gmail message is not in the permitted '{label_name}' label."
        )

    parsed = _parse_message(details)
    logger.info(
        "gmail_support_message_fetched label=%s message_id_present=%s",
        label_name,
        bool(parsed.get("id")),
    )
    return parsed


def fetch_support_emails(
    *,
    max_results: int = MAX_MESSAGES,
    label_name: str = LABEL_NAME,
    token_path: Path = TOKEN_FILE,
    service=None,
) -> list[dict[str, str]]:
    """
    Return up to `max_results` labeled support emails as structured dicts.

    Each item: id, sender, subject, body (plain text).
    """
    limit = max(1, min(int(max_results), MAX_MESSAGES))
    gmail = service
    if gmail is None:
        creds = load_gmail_credentials(token_path=token_path)
        gmail = _build_gmail_service(creds)

    label_id = _find_label_id(gmail, label_name=label_name)

    try:
        listed = (
            gmail.users()
            .messages()
            .list(userId="me", labelIds=[label_id], maxResults=limit)
            .execute()
        )
    except HttpError as exc:
        raise GmailApiError("Failed to list labeled Gmail messages.") from exc

    message_refs = listed.get("messages") or []
    emails: list[dict[str, str]] = []

    for ref in message_refs:
        message_id = ref.get("id")
        if not message_id:
            continue
        details = _get_message_details(gmail, message_id)
        emails.append(_parse_message(details))

    logger.info(
        "gmail_support_inbox_fetched count=%s label=%s",
        len(emails),
        label_name,
    )
    return emails
