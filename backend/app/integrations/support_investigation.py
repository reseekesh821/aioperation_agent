"""Investigate labeled support emails using existing read-only business tools."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from backend.app.agents.ai_errors import is_ai_service_error
from backend.app.agents.response_generation import (
    OLIST_FIELD_DELIVERY,
    OLIST_FIELD_FREIGHT,
    OLIST_FIELD_PAYMENT_VALUE,
    OLIST_FIELD_PRICE,
    OLIST_FIELD_STATUS,
    OLIST_INTERNAL_LEAK_MARKERS,
    detect_olist_requested_fields,
    render_olist_order_fallback,
)
from backend.app.agents.tools import format_olist_order_facts
from backend.app.agents.validation import (
    extract_olist_order_id,
    extract_order_id,
)
from backend.app.integrations.gmail_reader import fetch_support_email_by_id
from backend.app.services.inventory import (
    find_product_in_text,
    get_inventory,
    normalize_product_name,
)
from backend.app.services.olist_orders import get_olist_order
from backend.app.services.orders import get_order
from backend.app.services.refunds import check_refund_eligibility
from backend.app.services.vector_store import search_policies

logger = logging.getLogger(__name__)

INTENT_REFUND = "refund"
INTENT_ORDER_STATUS = "order_status"
INTENT_POLICY = "policy"
INTENT_INVENTORY = "inventory"
INTENT_OLIST = "olist_historical"
INTENT_UNKNOWN = "unknown"

TOPIC_STATUS = "status"
TOPIC_DELIVERY = "delivery"
TOPIC_PAYMENT = "payment"
TOPIC_SHIPPING = "shipping"
TOPIC_REFUND = "refund"
TOPIC_POLICY = "policy"
TOPIC_INVENTORY = "inventory"
TOPIC_PRICE = "price"

OWNERSHIP_UNCERTAINTY = (
    "Email sender ownership of the referenced order was not verified."
)

# Broader than chat routing only for support-email ID extraction.
_SUPPORT_ORDER_ID_PATTERN = re.compile(
    r"\border(?:\s*(?:id|number|#))?\s*[:#]?\s*(\d+)\b",
    re.IGNORECASE,
)

_INTERNAL_OPERATOR_MARKERS = OLIST_INTERNAL_LEAK_MARKERS + (
    "these facts must not be used",
    "must not be used to infer",
    "grounding",
    "langgraph",
    "tool output",
    "verified facts block",
)


def extract_support_order_id(text: str) -> int | None:
    """Extract a numeric Stage 1 order ID from untrusted support email text."""
    order_id = extract_order_id(text)
    if order_id is not None:
        return order_id
    match = _SUPPORT_ORDER_ID_PATTERN.search(text or "")
    if match is None:
        return None
    return int(match.group(1))


def extract_support_requested_topics(text: str) -> frozenset[str]:
    """
    Extract all supported customer question parts from untrusted email text.

    Multi-part inquiries (e.g. delivery + total payment) return multiple topics.
    """
    q = (text or "").casefold()
    topics: set[str] = set()

    if any(
        cue in q
        for cue in (
            "refund",
            "return",
            "money back",
            "reimburse",
            "reimbursement",
        )
    ):
        topics.add(TOPIC_REFUND)

    if any(
        cue in q
        for cue in (
            "warranty",
            "policy",
            "deadline",
            "store credit",
        )
    ):
        topics.add(TOPIC_POLICY)

    if any(
        cue in q
        for cue in (
            "status",
            "where is",
            "tracking",
            "shipment",
        )
    ):
        topics.add(TOPIC_STATUS)

    if any(
        cue in q
        for cue in (
            "deliver",
            "delivery confirmation",
            "confirm delivery",
            "was it delivered",
            "arrival",
        )
    ):
        topics.add(TOPIC_DELIVERY)
        topics.add(TOPIC_STATUS)

    if any(
        cue in q
        for cue in (
            "total payment",
            "total amount",
            "amount paid",
            "total paid",
            "payment total",
            "how much did i pay",
            "how much was paid",
            "how much i paid",
        )
    ):
        topics.add(TOPIC_PAYMENT)

    if any(
        cue in q
        for cue in (
            "shipping",
            "freight",
            "postage",
            "delivery fee",
            "shipping cost",
            "including shipping",
        )
    ):
        topics.add(TOPIC_SHIPPING)

    if "price" in q and "shipping" not in q:
        topics.add(TOPIC_PRICE)

    if find_product_in_text(text) is not None or any(
        cue in q for cue in ("inventory", "in stock", "how many", "available")
    ):
        topics.add(TOPIC_INVENTORY)

    # Olist field detector covers additional payment phrasing already used in chat.
    olist_fields = detect_olist_requested_fields(text)
    if OLIST_FIELD_STATUS in olist_fields:
        topics.add(TOPIC_STATUS)
    if OLIST_FIELD_DELIVERY in olist_fields:
        topics.add(TOPIC_DELIVERY)
        topics.add(TOPIC_STATUS)
    if OLIST_FIELD_PAYMENT_VALUE in olist_fields:
        topics.add(TOPIC_PAYMENT)
    if OLIST_FIELD_FREIGHT in olist_fields:
        topics.add(TOPIC_SHIPPING)
    if OLIST_FIELD_PRICE in olist_fields:
        topics.add(TOPIC_PRICE)

    return frozenset(topics)


def classify_support_intent(text: str) -> str:
    """Classify primary support intent from email text without trusting instructions."""
    topics = extract_support_requested_topics(text)
    q = (text or "").casefold()

    if TOPIC_REFUND in topics:
        return INTENT_REFUND

    if TOPIC_POLICY in topics and TOPIC_STATUS not in topics and TOPIC_PAYMENT not in topics:
        return INTENT_POLICY

    if (
        TOPIC_STATUS in topics
        or TOPIC_DELIVERY in topics
        or TOPIC_PAYMENT in topics
        or TOPIC_SHIPPING in topics
        or TOPIC_PRICE in topics
    ):
        return INTENT_ORDER_STATUS

    if TOPIC_INVENTORY in topics:
        return INTENT_INVENTORY

    if extract_olist_order_id(text) is not None:
        return INTENT_OLIST

    if extract_support_order_id(text) is not None:
        return INTENT_ORDER_STATUS

    if any(cue in q for cue in ("warranty", "policy", "deadline", "store credit")):
        return INTENT_POLICY

    return INTENT_UNKNOWN


def build_safe_investigation_question(
    intent: str,
    *,
    order_id: int | None = None,
    olist_order_id: str | None = None,
    product_name: str | None = None,
    requested_topics: frozenset[str] | set[str] | None = None,
) -> str | None:
    """
    Build a deterministic investigation question from extracted fields only.

    Raw email text is never forwarded into LangGraph. Multi-part topic sets produce
    a sanitized question that covers every supported customer ask.
    """
    topics = frozenset(requested_topics or ())

    if olist_order_id:
        if intent == INTENT_REFUND or TOPIC_REFUND in topics:
            return f"Is order {olist_order_id} eligible for a refund?"

        parts: list[str] = []
        if TOPIC_DELIVERY in topics or TOPIC_STATUS in topics:
            parts.append("delivery confirmation and status")
        if TOPIC_PAYMENT in topics:
            parts.append("total payment")
        if TOPIC_SHIPPING in topics:
            parts.append("shipping cost")
        if TOPIC_PRICE in topics and TOPIC_PAYMENT not in topics:
            parts.append("item price")

        if not parts:
            return f"What is the status of order {olist_order_id}?"
        if len(parts) == 1:
            return f"What is the {parts[0]} for order {olist_order_id}?"
        leading = ", ".join(parts[:-1])
        return (
            f"Confirm the {leading} and {parts[-1]} for order {olist_order_id}."
        )

    if order_id is not None:
        if intent == INTENT_REFUND or TOPIC_REFUND in topics:
            return f"Is order {order_id} eligible for a refund?"
        if intent == INTENT_POLICY or TOPIC_POLICY in topics:
            return f"What is the refund deadline for order {order_id}?"
        return f"What is the status of order {order_id}?"

    if intent == INTENT_POLICY or TOPIC_POLICY in topics:
        return "What is the refund deadline?"

    if (intent == INTENT_INVENTORY or TOPIC_INVENTORY in topics) and product_name:
        return f"How many {product_name} are available?"

    return None


def _policy_snippets(question: str = "What is the refund deadline?") -> str:
    policies = search_policies(question)
    if not policies:
        return "No relevant policy found."
    return "\n".join(policies)


def _collect_tool_evidence(
    intent: str,
    *,
    order_id: int | None,
    olist_order_id: str | None,
    product_name: str | None,
) -> tuple[str, str, list[str]]:
    """Reuse existing read-only services for structured evidence."""
    verified_parts: list[str] = []
    policy_information = ""
    uncertainties: list[str] = [OWNERSHIP_UNCERTAINTY]

    if olist_order_id:
        order = get_olist_order(olist_order_id)
        if order is None:
            verified_parts.append("Historical Olist order not found.")
            uncertainties.append(
                f"Historical Olist order {olist_order_id} could not be verified."
            )
        else:
            verified_parts.append(format_olist_order_facts(order))
        policy_information = (
            "Historical Olist records cannot establish current refund eligibility "
            "or authorize live refund actions."
        )
        return "\n".join(verified_parts), policy_information, uncertainties

    if order_id is not None:
        order = get_order(order_id)
        if order is None:
            verified_parts.append(f"Order {order_id} not found.")
            uncertainties.append(f"Order {order_id} was not found in live operations.")
        else:
            verified_parts.append(
                f"Order {order_id}: {order['product']} - {order['status']}."
            )
            if intent == INTENT_REFUND:
                refund = check_refund_eligibility(order_id)
                reason = refund.get("reason") or "Refund review unavailable"
                verified_parts.append(f"Order {order_id}: {reason}.")

        if intent in {INTENT_REFUND, INTENT_POLICY}:
            policy_information = _policy_snippets()
        return "\n".join(verified_parts), policy_information, uncertainties

    if intent == INTENT_POLICY:
        policy_information = _policy_snippets()
        uncertainties.append("No order ID was found in the email.")
        return "", policy_information, uncertainties

    if intent == INTENT_INVENTORY and product_name:
        canonical = normalize_product_name(product_name)
        quantity = get_inventory(product_name) if canonical else None
        if canonical is None or quantity is None:
            verified_parts.append(f"Product '{product_name}' is not in the catalog.")
        elif quantity == 0:
            verified_parts.append(f"{canonical}: 0 units available (out of stock).")
        else:
            verified_parts.append(f"{canonical}: {quantity} units available.")
        uncertainties.append("No order ID was found in the email.")
        return "\n".join(verified_parts), "", uncertainties

    uncertainties.append("No order ID was found in the email.")
    if intent == INTENT_UNKNOWN:
        uncertainties.append(
            "Support intent could not be determined from the email content."
        )
    return "", "", uncertainties


def _run_langgraph_summary(
    question: str,
    *,
    invoke_graph: Callable[[dict[str, Any]], Any],
) -> tuple[str | None, str | None]:
    """
    Invoke LangGraph with a sanitized question only.

    Returns (summary, error_detail).
    """
    try:
        result = invoke_graph({"messages": [("user", question)]})
        messages = result["messages"]
        summary = messages[-1].content if messages else None
        if isinstance(summary, str) and summary.strip():
            return summary.strip(), None
        return None, "Investigation summary was empty."
    except Exception as exc:  # noqa: BLE001
        if is_ai_service_error(exc):
            return None, "AI service is temporarily unavailable."
        logger.exception("support_investigation_langgraph_failed")
        return None, "Investigation summary could not be generated."


def _sanitize_operator_text(text: str | None) -> str:
    if not text:
        return ""
    kept: list[str] = []
    for line in text.splitlines():
        lower = line.casefold()
        if any(marker in lower for marker in _INTERNAL_OPERATOR_MARKERS):
            continue
        kept.append(line.rstrip())
    cleaned = "\n".join(line for line in kept if line.strip()).strip()
    # Drop leftover internal one-liners that were not line-scoped.
    lower = cleaned.casefold()
    if any(marker in lower for marker in _INTERNAL_OPERATOR_MARKERS):
        return ""
    return cleaned


def _olist_fact_line(verified_facts: str, label: str) -> str | None:
    match = re.search(
        rf"^{re.escape(label)}:\s*(.+)$",
        verified_facts,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match is None:
        return None
    value = match.group(1).strip()
    if not value or value.casefold() in {"none", "null", "n/a"}:
        return None
    return value


def build_operator_presentation(findings: dict[str, Any]) -> dict[str, Any]:
    """
    Operator-facing investigation sections without raw tool dumps or grounding rules.
    """
    verified = findings.get("verified_facts") or ""
    olist_order_id = findings.get("olist_order_id")
    order_id = findings.get("order_id")
    is_historical = bool(olist_order_id) or "olist order id:" in verified.casefold()

    status_lines: list[str] = []
    order_lines: list[str] = []
    payment_lines: list[str] = []

    if is_historical:
        status_value = _olist_fact_line(verified, "Historical status")
        delivered = _olist_fact_line(verified, "Delivered to customer")
        estimated = _olist_fact_line(verified, "Estimated delivery date")
        if status_value:
            status_lines.append(f"Status: {status_value}")
        if delivered:
            status_lines.append(f"Delivered to customer: {delivered}")
        elif "delivered to customer:" in verified.casefold():
            status_lines.append("No customer delivery date is recorded.")
        if estimated:
            status_lines.append(f"Estimated delivery date: {estimated}")

        prices = re.findall(r"\bprice=([0-9]+(?:\.[0-9]+)?)", verified, flags=re.I)
        freights = re.findall(r"\bfreight=([0-9]+(?:\.[0-9]+)?)", verified, flags=re.I)
        if prices:
            if len(prices) == 1:
                order_lines.append(f"Item price: {prices[0]}")
            else:
                order_lines.append(f"Item prices: {', '.join(prices)}")
        categories = re.findall(
            r"product category=([^\s(]+)",
            verified,
            flags=re.IGNORECASE,
        )
        if categories:
            order_lines.append(f"Product category: {categories[0]}")

        if freights:
            if len(freights) == 1:
                payment_lines.append(f"Shipping/freight: {freights[0]}")
            else:
                payment_lines.append(f"Shipping/freight amounts: {', '.join(freights)}")
        payments = re.findall(
            r"type=([a-z0-9_]+),\s*installments=(\d+),\s*value=([0-9]+(?:\.[0-9]+)?)",
            verified,
            flags=re.IGNORECASE,
        )
        for payment_type, installments, value in payments:
            payment_lines.append(
                f"Total payment: {value} "
                f"(method: {payment_type}, installments: {installments})"
            )
        if "historical olist order not found" in verified.casefold():
            status_lines.append("Historical Olist order not found.")
    else:
        cleaned_facts = _sanitize_operator_text(verified)
        if cleaned_facts:
            # Live Stage 1 facts are already short operator-readable lines.
            if ":" in cleaned_facts and " - " in cleaned_facts:
                status_lines.append(cleaned_facts)
            else:
                order_lines.append(cleaned_facts)

    summary = _sanitize_operator_text(findings.get("summary"))
    if not summary and is_historical and verified.strip():
        # Deterministic operator summary from verified facts + sanitized question.
        question = findings.get("safe_question") or ""
        summary = render_olist_order_fallback(verified, question)

    uncertainties = [
        item
        for item in (findings.get("uncertainties") or [])
        if isinstance(item, str) and item.strip()
    ]

    return {
        "data_scope": "historical" if is_historical else "live",
        "intent": findings.get("intent") or INTENT_UNKNOWN,
        "order_id": str(order_id) if order_id is not None else None,
        "olist_order_id": olist_order_id,
        "requested_topics": sorted(findings.get("requested_topics") or []),
        "status": "\n".join(status_lines).strip(),
        "order_details": "\n".join(order_lines).strip(),
        "payment_details": "\n".join(payment_lines).strip(),
        "policy_information": (findings.get("policy_information") or "").strip(),
        "uncertainties": uncertainties,
        "summary": summary,
    }


def investigate_support_email(
    message_id: str,
    *,
    fetch_email: Callable[..., dict[str, str]] = fetch_support_email_by_id,
    invoke_graph: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """
    Investigate one labeled support email and return structured findings.

    Email body is treated as untrusted data for extraction only.
    """
    email = fetch_email(message_id)
    subject = email.get("subject") or ""
    body = email.get("body") or ""
    # Untrusted extraction corpus — never passed through as agent instructions.
    corpus = f"{subject}\n{body}"

    intent = classify_support_intent(corpus)
    requested_topics = extract_support_requested_topics(corpus)
    order_id = extract_support_order_id(corpus)
    olist_order_id = extract_olist_order_id(corpus)
    product_name = find_product_in_text(corpus)

    # Prefer live Stage 1 numeric IDs when both forms appear.
    if order_id is not None:
        olist_order_id = None

    verified_facts, policy_information, uncertainties = _collect_tool_evidence(
        intent,
        order_id=order_id,
        olist_order_id=olist_order_id,
        product_name=product_name,
    )

    safe_question = build_safe_investigation_question(
        intent,
        order_id=order_id,
        olist_order_id=olist_order_id,
        product_name=product_name,
        requested_topics=requested_topics,
    )

    summary = None
    summary_error = None
    if safe_question:
        if invoke_graph is None:
            from backend.app.agents.agent_graph import agent_graph

            invoke_graph = agent_graph.invoke
        summary, summary_error = _run_langgraph_summary(
            safe_question,
            invoke_graph=invoke_graph,
        )
        if summary_error:
            uncertainties.append(summary_error)
    else:
        uncertainties.append(
            "No safe investigation question could be constructed from the email."
        )

    logger.info(
        "support_email_investigated intent=%s has_order_id=%s has_olist_id=%s topics=%s",
        intent,
        order_id is not None,
        bool(olist_order_id),
        sorted(requested_topics),
    )

    findings = {
        "message_id": email.get("id") or message_id,
        "intent": intent,
        "order_id": order_id,
        "olist_order_id": olist_order_id,
        "requested_topics": sorted(requested_topics),
        "verified_facts": verified_facts,
        "policy_information": policy_information,
        "uncertainties": uncertainties,
        "summary": summary,
        "safe_question": safe_question,
    }
    findings["presentation"] = build_operator_presentation(findings)
    return findings
