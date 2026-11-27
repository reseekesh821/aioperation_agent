"""PostgreSQL persistence for support cases (support schema only)."""

from __future__ import annotations

import os
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

CASE_STATUS_NEW = "new"
CASE_STATUS_INVESTIGATED = "investigated"
CASE_STATUS_DRAFT_READY = "draft_ready"
CASE_STATUS_DRAFT_SAVED = "draft_saved"
CASE_STATUS_NEEDS_ATTENTION = "needs_attention"
CASE_STATUS_CLOSED = "closed"

AUTO_STATUS_RANK = {
    CASE_STATUS_NEW: 0,
    CASE_STATUS_INVESTIGATED: 1,
    CASE_STATUS_DRAFT_READY: 2,
    CASE_STATUS_DRAFT_SAVED: 3,
}

MANUAL_STATUSES = frozenset({
    CASE_STATUS_NEEDS_ATTENTION,
    CASE_STATUS_CLOSED,
})

ALL_STATUSES = frozenset({
    CASE_STATUS_NEW,
    CASE_STATUS_INVESTIGATED,
    CASE_STATUS_DRAFT_READY,
    CASE_STATUS_DRAFT_SAVED,
    CASE_STATUS_NEEDS_ATTENTION,
    CASE_STATUS_CLOSED,
})


def _connect():
    return psycopg.connect(
        os.environ["DATABASE_URL"],
        row_factory=dict_row,
    )


def ensure_support_schema(*, conn=None) -> None:
    """
    Apply support schema statements idempotently (for tests / local bootstrap).

    Prefer infra/support_schema.sql via psql for operators; this mirrors it.
    """
    owns_connection = conn is None
    if owns_connection:
        conn = _connect()
    try:
        with conn:
            conn.execute("CREATE SCHEMA IF NOT EXISTS support")
            conn.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_type t
                        JOIN pg_namespace n ON n.oid = t.typnamespace
                        WHERE t.typname = 'case_status'
                          AND n.nspname = 'support'
                    ) THEN
                        CREATE TYPE support.case_status AS ENUM (
                            'new',
                            'investigated',
                            'draft_ready',
                            'draft_saved',
                            'needs_attention',
                            'closed'
                        );
                    END IF;
                END
                $$
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS support.cases (
                    case_id BIGSERIAL PRIMARY KEY,
                    gmail_message_id TEXT NOT NULL,
                    gmail_thread_id TEXT,
                    subject TEXT,
                    intent TEXT,
                    order_id INTEGER,
                    olist_order_id TEXT,
                    status support.case_status NOT NULL DEFAULT 'new',
                    gmail_draft_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT support_cases_gmail_message_id_key
                        UNIQUE (gmail_message_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS support.case_events (
                    event_id BIGSERIAL PRIMARY KEY,
                    case_id BIGINT NOT NULL
                        REFERENCES support.cases (case_id)
                        ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    detail TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT support_case_events_case_event_key
                        UNIQUE (case_id, event_key)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_support_cases_status
                    ON support.cases (status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_support_cases_updated_at
                    ON support.cases (updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_support_case_events_case_id_created
                    ON support.case_events (case_id, created_at ASC)
                """
            )
    finally:
        if owns_connection:
            conn.close()


def _serialize_case(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    status = payload.get("status")
    if status is not None:
        payload["status"] = str(status)
    for key in ("created_at", "updated_at"):
        value = payload.get(key)
        if value is not None and hasattr(value, "isoformat"):
            payload[key] = value.isoformat()
    return payload


def _serialize_event(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    created = payload.get("created_at")
    if created is not None and hasattr(created, "isoformat"):
        payload["created_at"] = created.isoformat()
    return payload


def _can_auto_advance(current: str, target: str) -> bool:
    if current in MANUAL_STATUSES:
        return False
    if target not in AUTO_STATUS_RANK:
        return False
    return AUTO_STATUS_RANK.get(current, -1) < AUTO_STATUS_RANK[target]


def record_event(
    conn,
    *,
    case_id: int,
    event_type: str,
    event_key: str,
    detail: str | None = None,
) -> bool:
    """Insert an idempotent case event. Returns True when a new row was created."""
    row = conn.execute(
        """
        INSERT INTO support.case_events (case_id, event_type, event_key, detail)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (case_id, event_key) DO NOTHING
        RETURNING event_id
        """,
        (case_id, event_type, event_key, detail),
    ).fetchone()
    return row is not None


def _normalize_optional_text(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    return cleaned or None


def upsert_case_from_gmail(
    *,
    gmail_message_id: str,
    gmail_thread_id: str | None = None,
    subject: str | None = None,
    intent: str | None = None,
    order_id: int | None = None,
    olist_order_id: str | None = None,
    source: str = "ensure",
    conn=None,
) -> dict[str, Any]:
    """
    Create or update a case from labeled Gmail metadata (no email body stored).

    Idempotent on gmail_message_id. Does not demote an advanced status.
    source=\"sync\" records sync events; source=\"ensure\" records case_created once.
    """
    message_id = (gmail_message_id or "").strip()
    if not message_id:
        raise ValueError("gmail_message_id is required")
    if source not in {"sync", "ensure"}:
        raise ValueError("source must be 'sync' or 'ensure'")

    thread_id = _normalize_optional_text(gmail_thread_id)
    subject = _normalize_optional_text(subject)
    intent = _normalize_optional_text(intent)
    olist_order_id = _normalize_optional_text(olist_order_id)

    owns_connection = conn is None
    if owns_connection:
        conn = _connect()

    try:
        with conn:
            with conn.transaction():
                existing = conn.execute(
                    """
                    SELECT * FROM support.cases
                    WHERE gmail_message_id = %s
                    FOR UPDATE
                    """,
                    (message_id,),
                ).fetchone()

                if existing is None:
                    row = conn.execute(
                        """
                        INSERT INTO support.cases (
                            gmail_message_id,
                            gmail_thread_id,
                            subject,
                            intent,
                            order_id,
                            olist_order_id,
                            status
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s::support.case_status)
                        RETURNING *
                        """,
                        (
                            message_id,
                            thread_id,
                            subject,
                            intent,
                            order_id,
                            olist_order_id,
                            CASE_STATUS_NEW,
                        ),
                    ).fetchone()
                    if source == "sync":
                        record_event(
                            conn,
                            case_id=row["case_id"],
                            event_type="synced",
                            event_key="synced",
                            detail="Case created from labeled Gmail sync.",
                        )
                    else:
                        record_event(
                            conn,
                            case_id=row["case_id"],
                            event_type="case_created",
                            event_key="case_created",
                            detail="Case created from support workflow.",
                        )
                    return _serialize_case(row)

                # Fill missing metadata only; never reset status back to new.
                conn.execute(
                    """
                    UPDATE support.cases
                    SET
                        gmail_thread_id = COALESCE(%s, gmail_thread_id),
                        subject = COALESCE(%s, subject),
                        intent = COALESCE(%s, intent),
                        order_id = COALESCE(%s, order_id),
                        olist_order_id = COALESCE(%s, olist_order_id),
                        updated_at = NOW()
                    WHERE case_id = %s
                    """,
                    (
                        thread_id,
                        subject,
                        intent,
                        order_id,
                        olist_order_id,
                        existing["case_id"],
                    ),
                )
                if source == "sync":
                    record_event(
                        conn,
                        case_id=existing["case_id"],
                        event_type="synced",
                        event_key="synced",
                        detail="Case metadata refreshed from labeled Gmail sync.",
                    )
                row = conn.execute(
                    "SELECT * FROM support.cases WHERE case_id = %s",
                    (existing["case_id"],),
                ).fetchone()
                return _serialize_case(row)
    finally:
        if owns_connection:
            conn.close()


def is_synthetic_gmail_message_id(message_id: str) -> bool:
    """
    Detect non-Gmail fixture IDs that leaked from localhost API tests.

    Real Gmail API message IDs are opaque hex-like tokens, not msg-* fixtures.
    """
    cleaned = (message_id or "").strip()
    if not cleaned:
        return True
    lower = cleaned.casefold()
    if lower.startswith("test-"):
        return True
    if lower.startswith("msg-"):
        return True
    if lower in {"msg-regression", "msg-outside", "msg-malicious"}:
        return True
    return False


def identify_inconsistent_support_cases(*, conn=None) -> dict[str, Any]:
    """Report synthetic / incomplete cases without mutating data."""
    owns_connection = conn is None
    if owns_connection:
        conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT case_id, gmail_message_id, subject, status, gmail_draft_id
            FROM support.cases
            ORDER BY case_id
            """
        ).fetchall()
        synthetic = []
        incomplete = []
        for row in rows:
            payload = _serialize_case(row)
            if is_synthetic_gmail_message_id(payload["gmail_message_id"]):
                synthetic.append(payload)
            elif not (payload.get("subject") or "").strip():
                incomplete.append(payload)
        return {
            "total": len(rows),
            "synthetic_cases": synthetic,
            "incomplete_cases": incomplete,
        }
    finally:
        if owns_connection:
            conn.close()


def reconcile_synthetic_support_cases(*, conn=None) -> dict[str, Any]:
    """
    Safely remove only synthetic fixture cases (msg-*/test-*).

    Preserves real Gmail message cases, their events, and draft references.
    """
    report = identify_inconsistent_support_cases(conn=conn)
    synthetic_ids = [
        case["gmail_message_id"] for case in report["synthetic_cases"]
    ]
    deleted = delete_cases_by_message_ids(synthetic_ids, conn=conn)
    return {
        "identified_synthetic": len(synthetic_ids),
        "deleted": deleted,
        "preserved_total_estimate": report["total"] - deleted,
        "synthetic_message_ids": synthetic_ids,
        "incomplete_real_cases": report["incomplete_cases"],
    }


def mark_case_investigated(
    *,
    gmail_message_id: str,
    intent: str | None = None,
    order_id: int | None = None,
    olist_order_id: str | None = None,
    summary: str | None = None,
    conn=None,
) -> dict[str, Any] | None:
    """Advance case to investigated after a successful investigation."""
    return _apply_auto_status(
        gmail_message_id=gmail_message_id,
        target_status=CASE_STATUS_INVESTIGATED,
        event_type="investigated",
        event_key="investigated",
        detail=(summary or "Investigation completed.")[:500],
        intent=intent,
        order_id=order_id,
        olist_order_id=olist_order_id,
        conn=conn,
    )


def mark_case_draft_ready(
    *,
    gmail_message_id: str,
    intent: str | None = None,
    order_id: int | None = None,
    olist_order_id: str | None = None,
    conn=None,
) -> dict[str, Any] | None:
    """Advance case to draft_ready after successful draft generation."""
    return _apply_auto_status(
        gmail_message_id=gmail_message_id,
        target_status=CASE_STATUS_DRAFT_READY,
        event_type="draft_ready",
        event_key="draft_ready",
        detail="Operator draft generated (not sent).",
        intent=intent,
        order_id=order_id,
        olist_order_id=olist_order_id,
        conn=conn,
    )


def mark_case_draft_saved(
    *,
    gmail_message_id: str,
    gmail_draft_id: str,
    conn=None,
) -> dict[str, Any] | None:
    """Store Gmail draft ID and advance to draft_saved only after Gmail success."""
    draft_id = (gmail_draft_id or "").strip()
    if not draft_id:
        raise ValueError("gmail_draft_id is required")

    owns_connection = conn is None
    if owns_connection:
        conn = _connect()

    try:
        with conn:
            with conn.transaction():
                existing = conn.execute(
                    """
                    SELECT * FROM support.cases
                    WHERE gmail_message_id = %s
                    FOR UPDATE
                    """,
                    (gmail_message_id,),
                ).fetchone()
                if existing is None:
                    return None

                current = str(existing["status"])
                next_status = (
                    CASE_STATUS_DRAFT_SAVED
                    if _can_auto_advance(current, CASE_STATUS_DRAFT_SAVED)
                    else current
                )
                conn.execute(
                    """
                    UPDATE support.cases
                    SET
                        gmail_draft_id = %s,
                        status = %s::support.case_status,
                        updated_at = NOW()
                    WHERE case_id = %s
                    """,
                    (draft_id, next_status, existing["case_id"]),
                )
                record_event(
                    conn,
                    case_id=existing["case_id"],
                    event_type="draft_saved",
                    event_key="draft_saved",
                    detail=f"Gmail draft saved (not sent). draft_id={draft_id}",
                )
                row = conn.execute(
                    "SELECT * FROM support.cases WHERE case_id = %s",
                    (existing["case_id"],),
                ).fetchone()
                return _serialize_case(row)
    finally:
        if owns_connection:
            conn.close()


def _apply_auto_status(
    *,
    gmail_message_id: str,
    target_status: str,
    event_type: str,
    event_key: str,
    detail: str | None,
    intent: str | None = None,
    order_id: int | None = None,
    olist_order_id: str | None = None,
    conn=None,
) -> dict[str, Any] | None:
    owns_connection = conn is None
    if owns_connection:
        conn = _connect()

    try:
        with conn:
            with conn.transaction():
                existing = conn.execute(
                    """
                    SELECT * FROM support.cases
                    WHERE gmail_message_id = %s
                    FOR UPDATE
                    """,
                    (gmail_message_id,),
                ).fetchone()
                if existing is None:
                    return None

                current = str(existing["status"])
                next_status = (
                    target_status
                    if _can_auto_advance(current, target_status)
                    else current
                )
                conn.execute(
                    """
                    UPDATE support.cases
                    SET
                        intent = COALESCE(%s, intent),
                        order_id = COALESCE(%s, order_id),
                        olist_order_id = COALESCE(%s, olist_order_id),
                        status = %s::support.case_status,
                        updated_at = NOW()
                    WHERE case_id = %s
                    """,
                    (
                        intent,
                        order_id,
                        olist_order_id,
                        next_status,
                        existing["case_id"],
                    ),
                )
                record_event(
                    conn,
                    case_id=existing["case_id"],
                    event_type=event_type,
                    event_key=event_key,
                    detail=detail,
                )
                row = conn.execute(
                    "SELECT * FROM support.cases WHERE case_id = %s",
                    (existing["case_id"],),
                ).fetchone()
                return _serialize_case(row)
    finally:
        if owns_connection:
            conn.close()


def set_manual_case_status(
    case_id: int,
    status: str,
    *,
    conn=None,
) -> dict[str, Any] | None:
    """Operator-only status updates: needs_attention or closed."""
    if status not in MANUAL_STATUSES:
        raise ValueError("Only needs_attention and closed may be set manually.")

    owns_connection = conn is None
    if owns_connection:
        conn = _connect()

    try:
        with conn:
            with conn.transaction():
                existing = conn.execute(
                    """
                    SELECT * FROM support.cases
                    WHERE case_id = %s
                    FOR UPDATE
                    """,
                    (case_id,),
                ).fetchone()
                if existing is None:
                    return None

                current = str(existing["status"])
                if current == status:
                    return _serialize_case(existing)

                conn.execute(
                    """
                    UPDATE support.cases
                    SET status = %s::support.case_status, updated_at = NOW()
                    WHERE case_id = %s
                    """,
                    (status, case_id),
                )
                # Unique key includes previous status so reopen/close history is kept
                # without duplicating identical no-op retries.
                record_event(
                    conn,
                    case_id=case_id,
                    event_type="manual_status",
                    event_key=f"manual:{current}->{status}",
                    detail=f"Operator set status to {status}.",
                )
                row = conn.execute(
                    "SELECT * FROM support.cases WHERE case_id = %s",
                    (case_id,),
                ).fetchone()
                return _serialize_case(row)
    finally:
        if owns_connection:
            conn.close()


def get_case(case_id: int, *, conn=None) -> dict[str, Any] | None:
    owns_connection = conn is None
    if owns_connection:
        conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM support.cases WHERE case_id = %s",
            (case_id,),
        ).fetchone()
        return _serialize_case(row)
    finally:
        if owns_connection:
            conn.close()


def get_case_by_message_id(
    gmail_message_id: str,
    *,
    conn=None,
) -> dict[str, Any] | None:
    owns_connection = conn is None
    if owns_connection:
        conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM support.cases WHERE gmail_message_id = %s",
            (gmail_message_id,),
        ).fetchone()
        return _serialize_case(row)
    finally:
        if owns_connection:
            conn.close()


def list_case_events(case_id: int, *, conn=None) -> list[dict[str, Any]]:
    owns_connection = conn is None
    if owns_connection:
        conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT event_id, case_id, event_type, event_key, detail, created_at
            FROM support.case_events
            WHERE case_id = %s
            ORDER BY created_at ASC, event_id ASC
            """,
            (case_id,),
        ).fetchall()
        return [_serialize_event(row) for row in rows]
    finally:
        if owns_connection:
            conn.close()


def list_cases(*, conn=None) -> list[dict[str, Any]]:
    owns_connection = conn is None
    if owns_connection:
        conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM support.cases
            ORDER BY updated_at DESC, case_id DESC
            """
        ).fetchall()
        return [_serialize_case(row) for row in rows]
    finally:
        if owns_connection:
            conn.close()


def count_cases_by_status(*, conn=None) -> dict[str, int]:
    owns_connection = conn is None
    if owns_connection:
        conn = _connect()
    try:
        counts = {status: 0 for status in ALL_STATUSES}
        rows = conn.execute(
            """
            SELECT status::text AS status, COUNT(*)::int AS count
            FROM support.cases
            GROUP BY status
            """
        ).fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["count"])
        return counts
    finally:
        if owns_connection:
            conn.close()


def delete_cases_by_message_ids(
    message_ids: Iterable[str],
    *,
    conn=None,
) -> int:
    """Test helper: remove cases (and cascaded events) by Gmail message ID."""
    ids = [mid for mid in message_ids if mid]
    if not ids:
        return 0
    owns_connection = conn is None
    if owns_connection:
        conn = _connect()
    try:
        with conn:
            result = conn.execute(
                "DELETE FROM support.cases WHERE gmail_message_id = ANY(%s)",
                (list(ids),),
            )
            return result.rowcount or 0
    finally:
        if owns_connection:
            conn.close()
