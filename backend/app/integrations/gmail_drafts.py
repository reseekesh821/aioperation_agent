"""Save operator-edited support replies as Gmail drafts (never send)."""

from __future__ import annotations

import base64
import logging
import re
from email.mime.text import MIMEText
from email.utils import parseaddr
from typing import Any, Callable

from googleapiclient.errors import HttpError

from backend.app.integrations.gmail_reader import (
    GmailApiError,
    GmailAuthError,
    LABEL_NAME,
    TOKEN_FILE,
    _build_gmail_service,
    fetch_support_email_by_id,
    load_gmail_credentials,
)

logger = logging.getLogger(__name__)

COMPOSE_SCOPE = "https://www.googleapis.com/auth/gmail.compose"
MAX_DRAFT_CHARS = 10_000
DRAFT_STATUS_SAVED_NOT_SENT = "saved_not_sent"

_EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)


class GmailDraftValidationError(Exception):
    """Edited draft text or derived recipient is invalid."""


def validate_draft_text(draft_text: str) -> str:
    cleaned = (draft_text or "").strip()
    if not cleaned:
        raise GmailDraftValidationError("Draft text cannot be empty.")
    if len(cleaned) > MAX_DRAFT_CHARS:
        raise GmailDraftValidationError(
            f"Draft text exceeds the {MAX_DRAFT_CHARS}-character limit."
        )
    return cleaned


def extract_email_address(header_value: str) -> str | None:
    """Extract a bare email address from a From/Reply-To header."""
    _name, address = parseaddr(header_value or "")
    address = (address or "").strip()
    if address and _EMAIL_PATTERN.fullmatch(address):
        return address
    # Fallback for bare addresses that parseaddr may not classify.
    candidate = (header_value or "").strip().strip("<>")
    if candidate and _EMAIL_PATTERN.fullmatch(candidate):
        return candidate
    return None


def validate_recipient_address(address: str | None) -> str:
    cleaned = (address or "").strip()
    if not cleaned or not _EMAIL_PATTERN.fullmatch(cleaned):
        raise GmailDraftValidationError(
            "Could not derive a valid recipient address from the original Gmail message."
        )
    return cleaned


def build_reply_subject(subject: str) -> str:
    cleaned = (subject or "").strip() or "(No subject)"
    if cleaned.casefold().startswith("re:"):
        return cleaned
    return f"Re: {cleaned}"


def derive_reply_recipient(email: dict[str, Any]) -> str:
    """Prefer Reply-To, otherwise From — never trust browser-supplied recipients."""
    for key in ("reply_to", "sender"):
        address = extract_email_address(email.get(key) or "")
        if address:
            return validate_recipient_address(address)
    raise GmailDraftValidationError(
        "Could not derive a valid recipient address from the original Gmail message."
    )


def build_reply_mime(
    *,
    draft_text: str,
    recipient: str,
    subject: str,
    rfc822_message_id: str = "",
    references: str = "",
) -> MIMEText:
    message = MIMEText(draft_text, "plain", "utf-8")
    message["To"] = recipient
    message["Subject"] = build_reply_subject(subject)

    message_id_header = (rfc822_message_id or "").strip()
    references_header = (references or "").strip()
    if message_id_header:
        message["In-Reply-To"] = message_id_header
        if references_header:
            message["References"] = f"{references_header} {message_id_header}".strip()
        else:
            message["References"] = message_id_header
    elif references_header:
        message["References"] = references_header

    return message


def _ensure_compose_scope(creds) -> None:
    scopes = set(getattr(creds, "scopes", None) or [])
    if scopes and COMPOSE_SCOPE not in scopes:
        raise GmailAuthError(
            "Gmail compose permission is missing. Re-run "
            "scripts/test_gmail_connection.py to authorize gmail.compose."
        )


def _create_gmail_draft(service, *, raw_message: str, thread_id: str = "") -> dict[str, Any]:
    body: dict[str, Any] = {"message": {"raw": raw_message}}
    if thread_id:
        body["message"]["threadId"] = thread_id

    try:
        return (
            service.users()
            .drafts()
            .create(userId="me", body=body)
            .execute()
        )
    except HttpError as exc:
        status = getattr(getattr(exc, "resp", None), "status", None)
        if status in {401, 403}:
            raise GmailAuthError(
                "Gmail authorization is missing compose permission or has expired. "
                "Re-run scripts/test_gmail_connection.py on localhost."
            ) from exc
        raise GmailApiError("Failed to create a Gmail draft.") from exc


def save_support_reply_draft(
    message_id: str,
    draft_text: str,
    *,
    fetch_email: Callable[..., dict[str, Any]] = fetch_support_email_by_id,
    service=None,
    token_path=TOKEN_FILE,
) -> dict[str, Any]:
    """
    Save an operator-edited reply as a Gmail draft for a labeled support email.

    Never sends mail. Recipient/subject/thread come from the original Gmail message.
    """
    cleaned_text = validate_draft_text(draft_text)

    gmail = service
    if gmail is None:
        creds = load_gmail_credentials(token_path=token_path)
        _ensure_compose_scope(creds)
        gmail = _build_gmail_service(creds)

    email = fetch_email(message_id, service=gmail)
    recipient = derive_reply_recipient(email)
    subject = build_reply_subject(email.get("subject") or "")
    thread_id = (email.get("thread_id") or "").strip()

    mime = build_reply_mime(
        draft_text=cleaned_text,
        recipient=recipient,
        subject=email.get("subject") or "",
        rfc822_message_id=email.get("rfc822_message_id") or "",
        references=email.get("references") or "",
    )
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")

    created = _create_gmail_draft(gmail, raw_message=raw, thread_id=thread_id)
    draft_id = created.get("id") or ""
    saved_thread = (
        ((created.get("message") or {}).get("threadId"))
        or thread_id
        or ""
    )

    logger.info(
        "gmail_draft_saved label=%s has_draft_id=%s has_thread=%s",
        LABEL_NAME,
        bool(draft_id),
        bool(saved_thread),
    )

    return {
        "message_id": email.get("id") or message_id,
        "draft_id": draft_id,
        "thread_id": saved_thread,
        "recipient": recipient,
        "subject": subject,
        "status": DRAFT_STATUS_SAVED_NOT_SENT,
        "sent": False,
    }
