"""Focused checks for separated customer/owner frontends and CORS."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import CORS_ALLOW_ORIGINS, app

ROOT = Path(__file__).resolve().parents[1]
CUSTOMER_DIR = ROOT / "frontend" / "customer"
OWNER_DIR = ROOT / "frontend" / "owner"

client = TestClient(app)


def test_frontend_entry_points_exist_as_server_roots():
    assert (CUSTOMER_DIR / "index.html").is_file()
    assert (CUSTOMER_DIR / "script.js").is_file()
    assert (CUSTOMER_DIR / "style.css").is_file()
    assert (OWNER_DIR / "index.html").is_file()
    assert (OWNER_DIR / "script.js").is_file()
    assert (OWNER_DIR / "style.css").is_file()
    # No shared static tree that becomes browsable when serving frontend/.
    assert not (ROOT / "frontend" / "shared").exists()
    assert not (ROOT / "frontend" / "index.html").exists()


def test_customer_portal_is_self_contained_chat_only():
    html = (CUSTOMER_DIR / "index.html").read_text(encoding="utf-8")
    script = (CUSTOMER_DIR / "script.js").read_text(encoding="utf-8")
    css = (CUSTOMER_DIR / "style.css").read_text(encoding="utf-8")

    assert 'id="chat-form"' in html
    assert 'href="style.css"' in html
    assert 'src="script.js"' in html
    assert "../shared/" not in html
    assert "support-inbox" not in html
    assert "support-cases" not in html
    assert "Load Emails" not in html
    assert "Save to Gmail Drafts" not in html

    assert 'const API_BASE = "http://127.0.0.1:8000"' in script
    assert "/chat" in script
    assert "/support/" not in script
    assert "#chat-box" in css


def test_owner_workspace_has_exclusive_inbox_and_cases_views():
    html = (OWNER_DIR / "index.html").read_text(encoding="utf-8")
    script = (OWNER_DIR / "script.js").read_text(encoding="utf-8")
    css = (OWNER_DIR / "style.css").read_text(encoding="utf-8")

    assert "Owner Support Workspace" in html
    assert 'id="support-inbox"' in html
    assert 'id="support-cases"' in html
    assert 'id="nav-inbox"' in html
    assert 'id="nav-cases"' in html
    assert 'data-view-panel="cases"' in html
    assert "hidden" in html
    assert 'href="style.css"' in html
    assert 'src="script.js"' in html
    assert "../shared/" not in html
    assert "chat-form" not in html
    assert "chat-box" not in html
    assert 'href="#support-inbox"' not in html
    assert 'href="#support-cases"' not in html

    assert "function showOwnerView" in script
    assert 'showOwnerView("inbox")' in script
    assert "inboxPanel.hidden" in script
    assert "casesPanel.hidden" in script
    assert "/support/emails" in script
    assert "/support/cases" in script
    assert "/chat" not in script
    assert "Save to Gmail Drafts" in script
    assert 'const API_BASE = "http://127.0.0.1:8000"' in script
    assert ".owner-nav-button" in css


def test_cors_allows_exact_ports_5000_and_5001_only():
    assert CORS_ALLOW_ORIGINS == [
        "http://127.0.0.1:5000",
        "http://127.0.0.1:5001",
    ]
    assert "*" not in CORS_ALLOW_ORIGINS
    assert "http://127.0.0.1:5500" not in CORS_ALLOW_ORIGINS
    assert "http://127.0.0.1:5501" not in CORS_ALLOW_ORIGINS

    allowed_customer = client.options(
        "/health",
        headers={
            "Origin": "http://127.0.0.1:5000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed_customer.status_code in {200, 204}
    assert (
        allowed_customer.headers.get("access-control-allow-origin")
        == "http://127.0.0.1:5000"
    )

    allowed_owner = client.options(
        "/health",
        headers={
            "Origin": "http://127.0.0.1:5001",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed_owner.status_code in {200, 204}
    assert (
        allowed_owner.headers.get("access-control-allow-origin")
        == "http://127.0.0.1:5001"
    )

    denied = client.options(
        "/health",
        headers={
            "Origin": "http://127.0.0.1:5500",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert denied.headers.get("access-control-allow-origin") is None
