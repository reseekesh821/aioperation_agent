-- Support Case Management schema (separate from public / olist).
-- Repeatable and non-destructive: safe to re-run against an existing volume.

CREATE SCHEMA IF NOT EXISTS support;

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
$$;

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
    CONSTRAINT support_cases_gmail_message_id_key UNIQUE (gmail_message_id)
);

CREATE TABLE IF NOT EXISTS support.case_events (
    event_id BIGSERIAL PRIMARY KEY,
    case_id BIGINT NOT NULL
        REFERENCES support.cases (case_id)
        ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    event_key TEXT NOT NULL,
    detail TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT support_case_events_case_event_key UNIQUE (case_id, event_key)
);

CREATE INDEX IF NOT EXISTS idx_support_cases_status
    ON support.cases (status);

CREATE INDEX IF NOT EXISTS idx_support_cases_updated_at
    ON support.cases (updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_support_case_events_case_id_created
    ON support.case_events (case_id, created_at ASC);
