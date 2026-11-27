from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend.app.integrations.gmail_reader import GmailApiError
from backend.app.integrations.support_draft import (
    generate_support_draft,
    generate_support_draft_text,
    is_support_draft_grounded,
    render_support_draft_fallback,
)
from backend.app.integrations.support_investigation import (
    OWNERSHIP_UNCERTAINTY,
    TOPIC_DELIVERY,
    TOPIC_PAYMENT,
    TOPIC_SHIPPING,
    TOPIC_STATUS,
)
from backend.app.main import app

client = TestClient(app)

OLIST_MULTI_PART_ORDER_ID = "0030d783f979fbc5981e75613b057344"


def _refund_findings():
    return {
        "message_id": "msg-1003",
        "intent": "refund",
        "order_id": 1003,
        "olist_order_id": None,
        "requested_topics": ["refund"],
        "verified_facts": (
            "Order 1003: Monitor - Delivered.\n"
            "Order 1003: Eligible for refund review."
        ),
        "policy_information": (
            "Standard refund requests must be submitted within 30 days after delivery."
        ),
        "uncertainties": [OWNERSHIP_UNCERTAINTY],
        "summary": "Order 1003 may be reviewed for a refund.",
        "safe_question": "Is order 1003 eligible for a refund?",
    }


def _olist_multi_part_findings():
    return {
        "message_id": "msg-olist-0030",
        "intent": "order_status",
        "order_id": None,
        "olist_order_id": OLIST_MULTI_PART_ORDER_ID,
        "requested_topics": [
            TOPIC_DELIVERY,
            TOPIC_PAYMENT,
            TOPIC_SHIPPING,
            TOPIC_STATUS,
        ],
        "verified_facts": (
            "Source: historical Olist ecommerce dataset (not a live operations order).\n"
            "These facts must not be used to infer current refund eligibility or live inventory.\n"
            f"Olist order ID: {OLIST_MULTI_PART_ORDER_ID}\n"
            "Historical status: delivered\n"
            "Purchase timestamp: 2017-12-01 00:00:00\n"
            "Delivered to customer: 2018-01-06 00:00:00\n"
            "Estimated delivery date: 2018-01-10 00:00:00\n"
            "Items (1):\n"
            "  - item 1: product category=toys, price=60.60, freight=17.67\n"
            "Payments (1):\n"
            "  - type=credit_card, installments=1, value=78.27\n"
            "No customer review was found for this order."
        ),
        "policy_information": (
            "Historical Olist records cannot establish current refund eligibility "
            "or authorize live refund actions."
        ),
        "uncertainties": [OWNERSHIP_UNCERTAINTY],
        "summary": (
            f"Historical Olist order {OLIST_MULTI_PART_ORDER_ID} has status delivered "
            "(delivered 2018-01-06). item price 60.60; shipping/freight 17.67; "
            "total amount paid 78.27."
        ),
        "safe_question": (
            f"Confirm the delivery confirmation and status, total payment and "
            f"shipping cost for order {OLIST_MULTI_PART_ORDER_ID}."
        ),
    }


class _GoodDraftLLM:
    def invoke(self, messages):
        return AIMessage(
            content=(
                "Thank you for contacting us about a refund for order 1003. "
                "Our records show that order 1003 is delivered and qualifies for "
                "refund review, which is not refund approval. Please reply with "
                "verification that you are the account holder for this order."
            )
        )


class _BadApprovalLLM:
    def invoke(self, messages):
        return AIMessage(
            content=(
                "Your refund for order 1003 has been approved and will be issued "
                "immediately. Ownership has been verified."
            )
        )


class _FailingLLM:
    def invoke(self, messages):
        raise RuntimeError("model crashed")


def test_refund_draft_fallback_covers_required_points():
    draft = render_support_draft_fallback(_refund_findings())
    lower = draft.casefold()
    assert "1003" in draft
    assert "delivered" in lower
    assert "refund review" in lower
    assert "approval" in lower
    assert "verify" in lower or "verification" in lower or "authorized" in lower
    assert "approved" not in lower


def test_grounded_model_draft_is_accepted():
    findings = _refund_findings()
    draft, used_fallback = generate_support_draft_text(findings, llm=_GoodDraftLLM())
    assert used_fallback is False
    assert is_support_draft_grounded(draft, findings)
    assert "1003" in draft
    assert "refund review" in draft.casefold()


def test_ungrounded_model_draft_falls_back():
    findings = _refund_findings()
    draft, used_fallback = generate_support_draft_text(findings, llm=_BadApprovalLLM())
    assert used_fallback is True
    assert "approved" not in draft.casefold()
    assert "refund review" in draft.casefold()
    assert "delivered" in draft.casefold()


def test_missing_evidence_uses_fallback_without_inventing_order():
    findings = {
        "message_id": "msg-none",
        "intent": "unknown",
        "order_id": None,
        "olist_order_id": None,
        "verified_facts": "",
        "policy_information": "",
        "uncertainties": [OWNERSHIP_UNCERTAINTY, "No order ID was found in the email."],
    }
    draft, used_fallback = generate_support_draft_text(findings, llm=_FailingLLM())
    assert used_fallback is True
    assert "order id" in draft.casefold()
    assert "1003" not in draft


def test_malicious_email_content_does_not_enter_draft(monkeypatch):
    findings = _refund_findings()

    def fake_investigate(message_id):
        return findings

    class JailbreakLLM:
        def invoke(self, messages):
            joined = " ".join(
                getattr(message, "content", "") for message in messages
            )
            assert "Ignore previous instructions" not in joined
            return AIMessage(
                content=(
                    "Thank you for contacting us about a refund for order 1003. "
                    "Order 1003 is delivered and qualifies for refund review, not "
                    "refund approval. Please verify that you are authorized on this "
                    "order."
                )
            )

    result = generate_support_draft(
        "msg-malicious",
        investigate=fake_investigate,
        llm=JailbreakLLM(),
    )
    assert result["status"] == "not_sent"
    assert "Ignore previous instructions" not in result["draft"]
    assert "approved" not in result["draft"].casefold()


def test_model_failure_falls_back_to_evidence_draft():
    findings = _refund_findings()
    draft, used_fallback = generate_support_draft_text(findings, llm=_FailingLLM())
    assert used_fallback is True
    assert "1003" in draft
    assert "delivered" in draft.casefold()
    assert "refund review" in draft.casefold()


class _GoodMultiPartLLM:
    def invoke(self, messages):
        return AIMessage(
            content=(
                f"Thank you for contacting support about order {OLIST_MULTI_PART_ORDER_ID}. "
                "Our historical records show the order was delivered on 2018-01-06. "
                "The item price was 60.60, shipping was 17.67, and the total payment "
                "was 78.27. Please reply with confirmation that you are authorized "
                "on the account for this order."
            )
        )


class _StatusOnlyLLM:
    def invoke(self, messages):
        return AIMessage(
            content=(
                f"Thank you for contacting support about order {OLIST_MULTI_PART_ORDER_ID}. "
                "Our historical records show the order was delivered on 2018-01-06. "
                "Please confirm that you are authorized on this account."
            )
        )


def test_multi_part_fallback_includes_delivery_and_payment():
    findings = _olist_multi_part_findings()
    draft = render_support_draft_fallback(findings)
    lower = draft.casefold()
    assert OLIST_MULTI_PART_ORDER_ID in draft
    assert "delivered" in lower
    assert "2018-01-06" in draft or "delivered" in lower
    assert "78.27" in draft
    assert "17.67" in draft
    assert "historical" in lower
    assert "authorized" in lower or "verify" in lower or "verification" in lower
    assert "approved" not in lower


def test_multi_part_model_draft_is_accepted():
    findings = _olist_multi_part_findings()
    draft, used_fallback = generate_support_draft_text(
        findings,
        llm=_GoodMultiPartLLM(),
    )
    assert used_fallback is False
    assert is_support_draft_grounded(draft, findings)
    assert "78.27" in draft
    assert "17.67" in draft
    assert "delivered" in draft.casefold()


def test_multi_part_incomplete_model_draft_falls_back():
    findings = _olist_multi_part_findings()
    draft, used_fallback = generate_support_draft_text(
        findings,
        llm=_StatusOnlyLLM(),
    )
    assert used_fallback is True
    assert "78.27" in draft
    assert "17.67" in draft
    assert "delivered" in draft.casefold()


def test_refund_1003_regression_still_covers_required_points():
    findings = _refund_findings()
    draft = render_support_draft_fallback(findings)
    assert "1003" in draft
    assert "delivered" in draft.casefold()
    assert "refund review" in draft.casefold()
    grounded, used_fallback = generate_support_draft_text(
        findings,
        llm=_GoodDraftLLM(),
    )
    assert used_fallback is False
    assert is_support_draft_grounded(grounded, findings)


def test_draft_endpoint_success(monkeypatch):
    monkeypatch.setattr(
        "backend.app.main.generate_support_draft",
        lambda message_id: {
            "message_id": message_id,
            "intent": "refund",
            "order_id": 1003,
            "olist_order_id": None,
            "draft": (
                "Thank you for contacting us about a refund for order 1003. "
                "Order 1003 is delivered and qualifies for refund review, not approval."
            ),
            "used_fallback": False,
            "status": "not_sent",
        },
    )
    monkeypatch.setattr("backend.app.main.record_draft_ready", lambda result: None)

    response = client.post("/support/emails/msg-1003/draft")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "not_sent"
    assert payload["order_id"] == 1003
    assert "refund review" in payload["draft"].casefold()
    assert "token" not in response.text.casefold()


def test_draft_endpoint_gmail_error(monkeypatch):
    def raise_gmail(message_id):
        raise GmailApiError("Failed to fetch a labeled Gmail message.")

    monkeypatch.setattr("backend.app.main.generate_support_draft", raise_gmail)
    monkeypatch.setattr("backend.app.main.record_draft_ready", lambda result: None)
    response = client.post("/support/emails/msg-1/draft")
    assert response.status_code == 502


def test_draft_endpoint_ollama_error(monkeypatch):
    def raise_ai(message_id):
        raise ConnectionError("ollama down")

    monkeypatch.setattr("backend.app.main.generate_support_draft", raise_ai)
    monkeypatch.setattr("backend.app.main.record_draft_ready", lambda result: None)
    response = client.post("/support/emails/msg-1/draft")
    assert response.status_code == 503
