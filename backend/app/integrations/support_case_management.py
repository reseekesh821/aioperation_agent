"""Orchestrate persistent support cases around existing Gmail/investigation/draft flows."""

from __future__ import annotations

import logging
from typing import Any, Callable

from backend.app.agents.validation import extract_olist_order_id
from backend.app.integrations.gmail_reader import fetch_support_emails
from backend.app.integrations.support_investigation import (
    classify_support_intent,
    extract_support_order_id,
)
from backend.app.services import support_cases as cases_db

logger = logging.getLogger(__name__)


def sync_support_cases(
    *,
    fetch_emails: Callable[..., list[dict[str, str]]] = fetch_support_emails,
) -> dict[str, Any]:
    """
    Explicitly sync labeled AI-Operations-Test Gmail messages into support.cases.

    Idempotent: repeated syncs do not create duplicate cases or duplicate sync events.
    Email bodies are used in-memory for extraction only and are never persisted.
    Sync fills missing subject/thread metadata and never resets a later status to new.
    """
    emails = fetch_emails()
    synced: list[dict[str, Any]] = []

    for email in emails:
        message_id = (email.get("id") or "").strip()
        if not message_id:
            continue

        subject = email.get("subject") or ""
        body = email.get("body") or ""
        # Untrusted extraction corpus — never stored on the case row.
        corpus = f"{subject}\n{body}"
        intent = classify_support_intent(corpus)
        order_id = extract_support_order_id(corpus)
        olist_order_id = extract_olist_order_id(corpus)
        if order_id is not None:
            olist_order_id = None

        case = cases_db.upsert_case_from_gmail(
            gmail_message_id=message_id,
            gmail_thread_id=email.get("thread_id") or None,
            subject=subject or None,
            intent=intent,
            order_id=order_id,
            olist_order_id=olist_order_id,
            source="sync",
        )
        synced.append(case)

    logger.info("support_cases_synced count=%s", len(synced))
    counts = cases_db.count_cases_by_status()
    all_cases = cases_db.list_cases()
    return {
        "synced_count": len(synced),
        "cases": all_cases,
        "counts": counts,
        "total": sum(counts.values()),
    }


def ensure_case_for_message(
    message_id: str,
    *,
    subject: str | None = None,
    thread_id: str | None = None,
    intent: str | None = None,
    order_id: int | None = None,
    olist_order_id: str | None = None,
) -> dict[str, Any]:
    """Ensure a case row exists before lifecycle updates (e.g. investigate without sync)."""
    return cases_db.upsert_case_from_gmail(
        gmail_message_id=message_id,
        gmail_thread_id=thread_id,
        subject=subject,
        intent=intent,
        order_id=order_id,
        olist_order_id=olist_order_id,
        source="ensure",
    )


def record_investigation_success(findings: dict[str, Any]) -> dict[str, Any] | None:
    """Update case after a successful investigation. Creates/reconciles by Gmail message ID."""
    message_id = (findings.get("message_id") or "").strip()
    if not message_id:
        return None

    ensure_case_for_message(
        message_id,
        subject=findings.get("subject"),
        thread_id=findings.get("thread_id"),
        intent=findings.get("intent"),
        order_id=findings.get("order_id"),
        olist_order_id=findings.get("olist_order_id"),
    )
    presentation = findings.get("presentation") or {}
    summary = presentation.get("summary") or findings.get("summary")
    return cases_db.mark_case_investigated(
        gmail_message_id=message_id,
        intent=findings.get("intent"),
        order_id=findings.get("order_id"),
        olist_order_id=findings.get("olist_order_id"),
        summary=summary if isinstance(summary, str) else None,
    )


def record_draft_ready(result: dict[str, Any]) -> dict[str, Any] | None:
    """Update case after successful draft generation (not sent)."""
    message_id = (result.get("message_id") or "").strip()
    if not message_id:
        return None

    ensure_case_for_message(
        message_id,
        subject=result.get("subject"),
        thread_id=result.get("thread_id"),
        intent=result.get("intent"),
        order_id=result.get("order_id"),
        olist_order_id=result.get("olist_order_id"),
    )
    return cases_db.mark_case_draft_ready(
        gmail_message_id=message_id,
        intent=result.get("intent"),
        order_id=result.get("order_id"),
        olist_order_id=result.get("olist_order_id"),
    )


def record_draft_saved(
    *,
    message_id: str,
    draft_id: str,
    subject: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any] | None:
    """Store draft ID only after Gmail drafts.create succeeded. Does not mark sent/closed."""
    message_id = (message_id or "").strip()
    draft_id = (draft_id or "").strip()
    if not message_id or not draft_id:
        return None

    ensure_case_for_message(
        message_id,
        subject=subject,
        thread_id=thread_id,
    )
    return cases_db.mark_case_draft_saved(
        gmail_message_id=message_id,
        gmail_draft_id=draft_id,
    )


def list_support_cases_dashboard() -> dict[str, Any]:
    cases = cases_db.list_cases()
    counts = cases_db.count_cases_by_status()
    return {
        "cases": cases,
        "counts": counts,
        "total": sum(counts.values()),
    }


def get_support_case_detail(case_id: int) -> dict[str, Any] | None:
    case = cases_db.get_case(case_id)
    if case is None:
        return None
    events = cases_db.list_case_events(case_id)
    return {
        "case": case,
        "events": events,
    }


def update_support_case_status(case_id: int, status: str) -> dict[str, Any]:
    return cases_db.set_manual_case_status(case_id, status)
