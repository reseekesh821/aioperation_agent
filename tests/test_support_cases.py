"""Persistent support case management tests (mocked Gmail/Ollama)."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app.integrations.support_case_management import (
    record_draft_ready,
    record_draft_saved,
    record_investigation_success,
    sync_support_cases,
)
from backend.app.main import app
from backend.app.services import support_cases as cases_db

client = TestClient(app)


def _unique_message_id(prefix: str = "case") -> str:
    return f"test-{prefix}-{uuid4().hex[:12]}"


@pytest.fixture(scope="module")
def support_schema():
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL is required for support case tests")
    cases_db.ensure_support_schema()
    yield


@pytest.fixture
def cleanup_message_ids(support_schema):
    ids: list[str] = []
    yield ids
    if ids:
        cases_db.delete_cases_by_message_ids(ids)


def _email(
    *,
    message_id: str,
    subject: str = "Refund request",
    body: str = "Please refund order 1003.",
    thread_id: str = "thread-1",
):
    return {
        "id": message_id,
        "thread_id": thread_id,
        "sender": "customer@example.com",
        "subject": subject,
        "body": body,
    }


def test_case_creation_and_persistence(cleanup_message_ids):
    message_id = _unique_message_id("create")
    cleanup_message_ids.append(message_id)

    case = cases_db.upsert_case_from_gmail(
        gmail_message_id=message_id,
        gmail_thread_id="thread-abc",
        subject="Where is order 1003?",
        intent="order_status",
        order_id=1003,
    )
    assert case["gmail_message_id"] == message_id
    assert case["status"] == "new"
    assert case["order_id"] == 1003
    assert case["subject"] == "Where is order 1003?"

    loaded = cases_db.get_case(case["case_id"])
    assert loaded is not None
    assert loaded["gmail_message_id"] == message_id
    assert loaded["gmail_thread_id"] == "thread-abc"


def test_gmail_message_deduplication_on_sync(cleanup_message_ids):
    message_id = _unique_message_id("dedupe")
    cleanup_message_ids.append(message_id)
    emails = [
        _email(
            message_id=message_id,
            subject="Delivery question",
            body="Confirm delivery for order 1003.",
        )
    ]

    first = sync_support_cases(fetch_emails=lambda: emails)
    second = sync_support_cases(fetch_emails=lambda: emails)

    assert first["synced_count"] == 1
    assert second["synced_count"] == 1

    matching = [
        case
        for case in cases_db.list_cases()
        if case["gmail_message_id"] == message_id
    ]
    assert len(matching) == 1

    events = cases_db.list_case_events(matching[0]["case_id"])
    synced_events = [event for event in events if event["event_type"] == "synced"]
    assert len(synced_events) == 1


def test_lifecycle_status_changes(cleanup_message_ids):
    message_id = _unique_message_id("life")
    cleanup_message_ids.append(message_id)
    cases_db.upsert_case_from_gmail(
        gmail_message_id=message_id,
        subject="Refund order 1003",
        intent="refund",
        order_id=1003,
    )

    investigated = record_investigation_success(
        {
            "message_id": message_id,
            "intent": "refund",
            "order_id": 1003,
            "olist_order_id": None,
            "summary": "Order 1003 may be reviewed for a refund.",
        }
    )
    assert investigated["status"] == "investigated"

    draft_ready = record_draft_ready(
        {
            "message_id": message_id,
            "intent": "refund",
            "order_id": 1003,
            "olist_order_id": None,
        }
    )
    assert draft_ready["status"] == "draft_ready"

    # Failed / missing draft ID must not mark draft_saved.
    assert (
        record_draft_saved(message_id=message_id, draft_id="") is None
    )
    still_ready = cases_db.get_case_by_message_id(message_id)
    assert still_ready["status"] == "draft_ready"
    assert still_ready["gmail_draft_id"] is None

    saved = record_draft_saved(message_id=message_id, draft_id="draft-xyz")
    assert saved["status"] == "draft_saved"
    assert saved["gmail_draft_id"] == "draft-xyz"


def test_failed_investigation_does_not_mark_success(cleanup_message_ids, monkeypatch):
    from backend.app.integrations.gmail_reader import GmailApiError

    message_id = _unique_message_id("fail-inv")
    cleanup_message_ids.append(message_id)
    cases_db.upsert_case_from_gmail(
        gmail_message_id=message_id,
        subject="Help",
        intent="unknown",
    )

    def boom(_message_id):
        raise GmailApiError("Failed to fetch a labeled Gmail message.")

    monkeypatch.setattr("backend.app.main.investigate_support_email", boom)
    response = client.post(f"/support/emails/{message_id}/investigate")
    assert response.status_code == 502

    case = cases_db.get_case_by_message_id(message_id)
    assert case["status"] == "new"
    events = cases_db.list_case_events(case["case_id"])
    assert not any(event["event_type"] == "investigated" for event in events)


def test_failed_draft_does_not_mark_draft_ready(cleanup_message_ids, monkeypatch):
    message_id = _unique_message_id("fail-draft")
    cleanup_message_ids.append(message_id)
    cases_db.upsert_case_from_gmail(
        gmail_message_id=message_id,
        subject="Help",
        intent="refund",
        order_id=1003,
    )
    record_investigation_success(
        {
            "message_id": message_id,
            "intent": "refund",
            "order_id": 1003,
            "summary": "Investigated.",
        }
    )

    def boom(_message_id):
        raise ConnectionError("ollama down")

    monkeypatch.setattr("backend.app.main.generate_support_draft", boom)
    response = client.post(f"/support/emails/{message_id}/draft")
    assert response.status_code == 503

    case = cases_db.get_case_by_message_id(message_id)
    assert case["status"] == "investigated"
    events = cases_db.list_case_events(case["case_id"])
    assert not any(event["event_type"] == "draft_ready" for event in events)


def test_draft_id_saved_only_after_gmail_success(cleanup_message_ids, monkeypatch):
    message_id = _unique_message_id("save")
    cleanup_message_ids.append(message_id)
    cases_db.upsert_case_from_gmail(
        gmail_message_id=message_id,
        subject="Refund",
        intent="refund",
        order_id=1003,
    )
    record_investigation_success(
        {
            "message_id": message_id,
            "intent": "refund",
            "order_id": 1003,
            "summary": "ok",
        }
    )
    record_draft_ready(
        {
            "message_id": message_id,
            "intent": "refund",
            "order_id": 1003,
        }
    )

    from backend.app.integrations.gmail_drafts import GmailApiError

    def fail_save(mid, draft_text):
        raise GmailApiError("Failed to create a Gmail draft.")

    monkeypatch.setattr("backend.app.main.save_support_reply_draft", fail_save)
    response = client.post(
        f"/support/emails/{message_id}/save-draft",
        json={"draft_text": "Edited reply"},
    )
    assert response.status_code == 502
    case = cases_db.get_case_by_message_id(message_id)
    assert case["status"] == "draft_ready"
    assert case["gmail_draft_id"] is None

    monkeypatch.setattr(
        "backend.app.main.save_support_reply_draft",
        lambda mid, draft_text: {
            "message_id": mid,
            "draft_id": "draft-777",
            "thread_id": "thread-1",
            "recipient": "customer@example.com",
            "subject": "Re: Refund",
            "status": "saved_not_sent",
            "sent": False,
        },
    )
    ok = client.post(
        f"/support/emails/{message_id}/save-draft",
        json={"draft_text": "Edited reply"},
    )
    assert ok.status_code == 200
    case = cases_db.get_case_by_message_id(message_id)
    assert case["status"] == "draft_saved"
    assert case["gmail_draft_id"] == "draft-777"


def test_manual_status_updates_and_event_history(cleanup_message_ids):
    message_id = _unique_message_id("manual")
    cleanup_message_ids.append(message_id)
    case = cases_db.upsert_case_from_gmail(
        gmail_message_id=message_id,
        subject="Manual path",
        intent="unknown",
    )
    case_id = case["case_id"]

    attention = client.post(
        f"/support/cases/{case_id}/status",
        json={"status": "needs_attention"},
    )
    assert attention.status_code == 200
    assert attention.json()["case"]["status"] == "needs_attention"

    # Repeated identical request is a no-op without a duplicate event.
    again = client.post(
        f"/support/cases/{case_id}/status",
        json={"status": "needs_attention"},
    )
    assert again.status_code == 200

    closed = client.post(
        f"/support/cases/{case_id}/status",
        json={"status": "closed"},
    )
    assert closed.status_code == 200
    assert closed.json()["case"]["status"] == "closed"

    detail = client.get(f"/support/cases/{case_id}")
    assert detail.status_code == 200
    payload = detail.json()
    events = payload["events"]
    manual = [event for event in events if event["event_type"] == "manual_status"]
    assert len(manual) == 2
    assert payload["case"]["status"] == "closed"

    bad = client.post(
        f"/support/cases/{case_id}/status",
        json={"status": "draft_saved"},
    )
    assert bad.status_code == 422


def test_list_cases_endpoint_and_sync_endpoint(cleanup_message_ids, monkeypatch):
    message_id = _unique_message_id("api")
    cleanup_message_ids.append(message_id)

    monkeypatch.setattr(
        "backend.app.integrations.support_case_management.fetch_support_emails",
        lambda: [
            _email(
                message_id=message_id,
                subject="Order status for 1003",
                body="What is the status of order 1003?",
            )
        ],
    )
    # Patch the default used by sync_support_cases when called via main
    monkeypatch.setattr(
        "backend.app.main.sync_support_cases",
        lambda: sync_support_cases(
            fetch_emails=lambda: [
                _email(
                    message_id=message_id,
                    subject="Order status for 1003",
                    body="What is the status of order 1003?",
                )
            ]
        ),
    )

    sync = client.post("/support/cases/sync")
    assert sync.status_code == 200
    body = sync.json()
    assert body["synced_count"] == 1
    assert body["total"] >= 1
    assert "new" in body["counts"]

    listed = client.get("/support/cases")
    assert listed.status_code == 200
    listed_body = listed.json()
    assert listed_body["total"] >= 1
    assert any(
        case["gmail_message_id"] == message_id for case in listed_body["cases"]
    )


def test_investigate_endpoint_updates_case_on_success(
    cleanup_message_ids,
    monkeypatch,
):
    message_id = _unique_message_id("inv-ok")
    cleanup_message_ids.append(message_id)
    cases_db.upsert_case_from_gmail(
        gmail_message_id=message_id,
        subject="Refund order 1003",
        intent="refund",
        order_id=1003,
    )

    monkeypatch.setattr(
        "backend.app.main.investigate_support_email",
        lambda mid: {
            "message_id": mid,
            "intent": "refund",
            "order_id": 1003,
            "olist_order_id": None,
            "requested_topics": ["refund"],
            "verified_facts": "Order 1003: Monitor - Delivered.",
            "policy_information": "Within 30 days.",
            "uncertainties": ["ownership unverified"],
            "summary": "Order 1003 can be reviewed for a refund.",
            "presentation": {
                "data_scope": "live",
                "status": "Delivered",
                "order_details": "",
                "payment_details": "",
                "policy_information": "Within 30 days.",
                "uncertainties": ["ownership unverified"],
                "summary": "Order 1003 can be reviewed for a refund.",
            },
        },
    )

    response = client.post(f"/support/emails/{message_id}/investigate")
    assert response.status_code == 200
    case = cases_db.get_case_by_message_id(message_id)
    assert case["status"] == "investigated"
    events = cases_db.list_case_events(case["case_id"])
    assert any(event["event_type"] == "investigated" for event in events)

    # Repeated success does not duplicate the investigated event.
    response2 = client.post(f"/support/emails/{message_id}/investigate")
    assert response2.status_code == 200
    events2 = cases_db.list_case_events(case["case_id"])
    assert len([e for e in events2 if e["event_type"] == "investigated"]) == 1


def test_draft_endpoint_regression_still_returns_not_sent(monkeypatch):
    monkeypatch.setattr(
        "backend.app.main.generate_support_draft",
        lambda message_id: {
            "message_id": message_id,
            "intent": "refund",
            "order_id": 1003,
            "olist_order_id": None,
            "draft": "Thank you for contacting us about order 1003.",
            "used_fallback": False,
            "status": "not_sent",
        },
    )
    monkeypatch.setattr(
        "backend.app.main.record_draft_ready",
        lambda result: None,
    )
    response = client.post("/support/emails/msg-regression/draft")
    assert response.status_code == 200
    assert response.json()["status"] == "not_sent"


def test_sync_before_investigation_preserves_one_case(cleanup_message_ids):
    message_id = _unique_message_id("sync-first")
    cleanup_message_ids.append(message_id)
    emails = [
        _email(
            message_id=message_id,
            subject="Refund request for order 1003",
            body="Please refund order 1003.",
            thread_id="thread-sync-first",
        )
    ]
    sync_support_cases(fetch_emails=lambda: emails)
    case = record_investigation_success(
        {
            "message_id": message_id,
            "subject": "Refund request for order 1003",
            "thread_id": "thread-sync-first",
            "intent": "refund",
            "order_id": 1003,
            "summary": "Order 1003 may be reviewed for a refund.",
        }
    )
    assert case["status"] == "investigated"
    assert case["subject"] == "Refund request for order 1003"
    matching = [
        row for row in cases_db.list_cases() if row["gmail_message_id"] == message_id
    ]
    assert len(matching) == 1
    events = cases_db.list_case_events(case["case_id"])
    assert len([e for e in events if e["event_type"] == "synced"]) == 1
    assert len([e for e in events if e["event_type"] == "investigated"]) == 1


def test_investigation_before_sync_fills_subject_without_reset(cleanup_message_ids):
    message_id = _unique_message_id("inv-first")
    cleanup_message_ids.append(message_id)

    case = record_investigation_success(
        {
            "message_id": message_id,
            "subject": "Delivery confirmation needed",
            "thread_id": "thread-inv-first",
            "intent": "order_status",
            "order_id": 1003,
            "summary": "Order 1003 is delivered.",
        }
    )
    assert case["status"] == "investigated"
    assert case["subject"] == "Delivery confirmation needed"
    created_events = [
        e
        for e in cases_db.list_case_events(case["case_id"])
        if e["event_type"] == "case_created"
    ]
    assert len(created_events) == 1

    sync_support_cases(
        fetch_emails=lambda: [
            _email(
                message_id=message_id,
                subject="Delivery confirmation needed",
                body="Confirm delivery for order 1003.",
                thread_id="thread-inv-first",
            )
        ]
    )
    refreshed = cases_db.get_case_by_message_id(message_id)
    assert refreshed["status"] == "investigated"
    assert refreshed["subject"] == "Delivery confirmation needed"
    assert refreshed["gmail_thread_id"] == "thread-inv-first"
    matching = [
        row for row in cases_db.list_cases() if row["gmail_message_id"] == message_id
    ]
    assert len(matching) == 1


def test_draft_saving_before_sync_reconciles_metadata(cleanup_message_ids):
    message_id = _unique_message_id("draft-first")
    cleanup_message_ids.append(message_id)

    # Simulate draft-save path creating the case before inbox sync.
    saved = record_draft_saved(
        message_id=message_id,
        draft_id="draft-abc",
        subject=None,
        thread_id=None,
    )
    assert saved["status"] == "draft_saved"
    assert saved["gmail_draft_id"] == "draft-abc"
    assert not (saved.get("subject") or "").strip()

    sync_support_cases(
        fetch_emails=lambda: [
            _email(
                message_id=message_id,
                subject="Order Delivery Inquiry",
                body="Confirm delivery and total payment.",
                thread_id="thread-draft-first",
            )
        ]
    )
    reconciled = cases_db.get_case_by_message_id(message_id)
    assert reconciled["status"] == "draft_saved"
    assert reconciled["gmail_draft_id"] == "draft-abc"
    assert reconciled["subject"] == "Order Delivery Inquiry"
    assert reconciled["gmail_thread_id"] == "thread-draft-first"
    events = cases_db.list_case_events(reconciled["case_id"])
    assert len([e for e in events if e["event_type"] == "draft_saved"]) == 1
    assert len([e for e in events if e["event_type"] == "synced"]) == 1
    assert len(
        [
            row
            for row in cases_db.list_cases()
            if row["gmail_message_id"] == message_id
        ]
    ) == 1


def test_repeated_sync_preserves_advanced_status_and_events(cleanup_message_ids):
    message_id = _unique_message_id("repeat-sync")
    cleanup_message_ids.append(message_id)
    emails = [
        _email(
            message_id=message_id,
            subject="Refund request for order 1003",
            body="Please refund order 1003.",
        )
    ]
    sync_support_cases(fetch_emails=lambda: emails)
    record_investigation_success(
        {
            "message_id": message_id,
            "subject": "Refund request for order 1003",
            "intent": "refund",
            "order_id": 1003,
            "summary": "Investigated.",
        }
    )
    record_draft_ready(
        {
            "message_id": message_id,
            "subject": "Refund request for order 1003",
            "intent": "refund",
            "order_id": 1003,
        }
    )
    sync_support_cases(fetch_emails=lambda: emails)
    sync_support_cases(fetch_emails=lambda: emails)

    case = cases_db.get_case_by_message_id(message_id)
    assert case["status"] == "draft_ready"
    assert case["subject"] == "Refund request for order 1003"
    events = cases_db.list_case_events(case["case_id"])
    assert len([e for e in events if e["event_type"] == "synced"]) == 1
    assert len([e for e in events if e["event_type"] == "investigated"]) == 1
    assert len([e for e in events if e["event_type"] == "draft_ready"]) == 1


def test_identify_and_reconcile_synthetic_cases(cleanup_message_ids):
    real_id = f"a{uuid4().hex[:15]}"
    synthetic_id = "msg-pollution-fixture"
    cleanup_message_ids.extend([real_id, synthetic_id])

    cases_db.upsert_case_from_gmail(
        gmail_message_id=real_id,
        subject="Real Gmail case",
        source="sync",
    )
    cases_db.upsert_case_from_gmail(
        gmail_message_id=synthetic_id,
        subject=None,
        source="ensure",
    )
    report = cases_db.identify_inconsistent_support_cases()
    assert any(
        case["gmail_message_id"] == synthetic_id
        for case in report["synthetic_cases"]
    )

    result = cases_db.reconcile_synthetic_support_cases()
    assert synthetic_id in result["synthetic_message_ids"]
    assert cases_db.get_case_by_message_id(synthetic_id) is None
    assert cases_db.get_case_by_message_id(real_id) is not None
    # Already deleted by reconcile; avoid double-delete noise in fixture cleanup.
    cleanup_message_ids.remove(synthetic_id)
