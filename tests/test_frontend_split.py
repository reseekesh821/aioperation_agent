"""Focused checks for separated customer/owner frontends and CORS."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import CORS_ALLOW_ORIGINS, app

ROOT = Path(__file__).resolve().parents[1]
CUSTOMER_DIR = ROOT / "frontend" / "customer"
OWNER_DIR = ROOT / "frontend" / "owner"
SHARED_DIR = ROOT / "frontend" / "shared"

client = TestClient(app)


def test_frontend_entry_points_exist():
    assert (CUSTOMER_DIR / "index.html").is_file()
    assert (CUSTOMER_DIR / "script.js").is_file()
    assert (CUSTOMER_DIR / "style.css").is_file()
    assert (OWNER_DIR / "index.html").is_file()
    assert (OWNER_DIR / "script.js").is_file()
    assert (OWNER_DIR / "style.css").is_file()
    assert (SHARED_DIR / "config.js").is_file()


def test_customer_portal_exposes_chat_only():
    html = (CUSTOMER_DIR / "index.html").read_text(encoding="utf-8")
    script = (CUSTOMER_DIR / "script.js").read_text(encoding="utf-8")

    assert 'id="chat-form"' in html
    assert "AI Operations Agent" in html
    assert "support-inbox" not in html
    assert "support-cases" not in html
    assert "Load Emails" not in html
    assert "Save to Gmail Drafts" not in html
    assert "Sync from Gmail" not in html

    assert "/chat" in script
    assert "/support/" not in script
    assert "AIOPS_API_BASE" in script or "127.0.0.1:8000" in script


def test_owner_workspace_exposes_inbox_and_cases_without_chat():
    html = (OWNER_DIR / "index.html").read_text(encoding="utf-8")
    script = (OWNER_DIR / "script.js").read_text(encoding="utf-8")

    assert "Owner Support Workspace" in html
    assert 'id="support-inbox"' in html
    assert 'id="support-cases"' in html
    assert 'href="#support-inbox"' in html
    assert 'href="#support-cases"' in html
    assert "chat-form" not in html
    assert "chat-box" not in html
    assert "not authentication" in html.casefold()

    assert "/support/emails" in script
    assert "/support/cases" in script
    assert "/chat" not in script
    assert "Save to Gmail Drafts" in script
    assert "AIOPS_API_BASE" in script or "127.0.0.1:8000" in script


def test_shared_config_targets_localhost_backend():
    config = (SHARED_DIR / "config.js").read_text(encoding="utf-8")
    assert "http://127.0.0.1:8000" in config
    assert "token" not in config.casefold()
    assert "credential" not in config.casefold()


def test_cors_allows_exact_customer_and_owner_origins_only():
    assert CORS_ALLOW_ORIGINS == [
        "http://127.0.0.1:5500",
        "http://127.0.0.1:5501",
    ]
    assert "*" not in CORS_ALLOW_ORIGINS

    allowed_customer = client.options(
        "/health",
        headers={
            "Origin": "http://127.0.0.1:5500",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed_customer.status_code in {200, 204}
    assert (
        allowed_customer.headers.get("access-control-allow-origin")
        == "http://127.0.0.1:5500"
    )

    allowed_owner = client.options(
        "/health",
        headers={
            "Origin": "http://127.0.0.1:5501",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed_owner.status_code in {200, 204}
    assert (
        allowed_owner.headers.get("access-control-allow-origin")
        == "http://127.0.0.1:5501"
    )

    denied = client.options(
        "/health",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert denied.headers.get("access-control-allow-origin") is None


def test_old_combined_frontend_root_files_removed():
    frontend_root = ROOT / "frontend"
    assert not (frontend_root / "index.html").exists()
    assert not (frontend_root / "script.js").exists()
    assert not (frontend_root / "style.css").exists()
