import pytest
import httpx
from backend.app.agents.response_generation import (
    GeneratedResponse,
    extract_required_anchors,
    generate_grounded_response,
    get_last_generation,
    is_response_grounded,
)
from backend.app.services.policy_evidence import APPROVED_CLAIMS, NO_POLICY_INFO


class ScriptedLLM:
    """Deterministic stand-in that returns queued candidate texts."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        content = self.replies.pop(0) if self.replies else ""

        class Resp:
            pass

        resp = Resp()
        resp.content = content
        return resp


def test_extract_inventory_anchors():
    facts = "Monitor: 5 units available."
    anchors = extract_required_anchors(facts)
    assert "5" in anchors["numbers"]
    assert "monitor" in anchors["products"]


def test_extract_order_anchors():
    facts = "Order 1003: Mouse - Delivered."
    anchors = extract_required_anchors(facts)
    assert "1003" in anchors["numbers"]
    assert "delivered" in anchors["statuses"]
    assert "mouse" in anchors["products"]


def test_grounded_inventory_paraphrase_accepted():
    facts = "Monitor: 5 units available."
    answer = "We currently have 5 Monitor units available."
    assert is_response_grounded(answer, facts)


def test_zero_stock_natural_without_digit():
    facts = "Mouse: 0 units available (out of stock)."
    answer = "We're currently out of stock on mice."
    assert is_response_grounded(answer, facts)


def test_order_status_without_product_accepted():
    facts = "Order 1003: Mouse - Delivered."
    answer = "The status of order 1003 is Delivered."
    assert is_response_grounded(answer, facts)


def test_rejects_unsupported_product_attribute():
    facts = "Order 1003: Mouse - Delivered."
    answer = (
        "Order 1003 has been delivered. "
        "The order was for a standard mouse."
    )
    assert not is_response_grounded(answer, facts)


def test_rejects_invented_product_specification():
    facts = "Order 1003: Mouse - Delivered."
    answer = "Order 1003 included a wireless gaming mouse and is Delivered."
    assert not is_response_grounded(answer, facts)


def test_accepts_verified_product_without_extra_attributes():
    facts = "Order 1003: Mouse - Delivered."
    answer = "Order 1003 has been delivered. The order was for a mouse."
    assert is_response_grounded(answer, facts)


def test_accepts_attribute_present_in_verified_facts():
    facts = "Order 1001: Wireless Mouse - Delivered."
    answer = "Order 1001 has been delivered and was for a wireless mouse."
    assert is_response_grounded(answer, facts)


def test_generate_falls_back_on_unsupported_product_attribute():
    facts = "Order 1003: Mouse - Delivered."
    llm = ScriptedLLM([
        "Order 1003 has been delivered. The order was for a standard mouse.",
        "Order 1003 shipped a premium ergonomic mouse that is Delivered.",
    ])
    result = generate_grounded_response(
        "What is the status of order 1003?",
        facts,
        llm,
    )
    assert result.used_fallback is True
    assert result.answer == facts
    assert "standard" not in result.answer.lower()


def test_rejects_changed_quantity():
    facts = "Monitor: 5 units available."
    answer = "We currently have 12 Monitor units available."
    assert not is_response_grounded(answer, facts)


def test_rejects_unsupported_policy_claim():
    facts = APPROVED_CLAIMS["refund_1"]
    answer = "You can get store credit within 30 days after delivery."
    assert not is_response_grounded(answer, facts)


def test_rejects_weak_answer_missing_standard_and_facts():
    facts = APPROVED_CLAIMS["refund_3"]
    answer = "Refunds are fine anytime."
    assert not is_response_grounded(answer, facts)


def test_accepts_deadline_paraphrase_without_standard():
    facts = APPROVED_CLAIMS["refund_1"]
    answer = "The refund deadline is 30 days after delivery."
    assert is_response_grounded(answer, facts)


def test_accepts_natural_no_policy_info():
    answer = "Our available policies do not specify this information."
    assert is_response_grounded(answer, NO_POLICY_INFO)


def test_rejects_no_policy_info_with_invention():
    answer = "Yes, we offer a lifetime warranty."
    assert not is_response_grounded(answer, NO_POLICY_INFO)


def test_generate_accepts_valid_candidate():
    facts = "Monitor: 5 units available."
    llm = ScriptedLLM([
        "There are 5 Monitor units available right now."
    ])
    result = generate_grounded_response(
        "How many monitors are available?",
        facts,
        llm,
    )
    assert isinstance(result, GeneratedResponse)
    assert result.used_fallback is False
    assert result.attempts == 1
    assert "5" in result.answer
    assert get_last_generation() is result


def test_generate_retries_then_falls_back():
    facts = "Monitor: 5 units available."
    llm = ScriptedLLM([
        "We have 99 laptops in a lifetime warranty sale.",
        "Unlimited Monitor stock with store credit.",
    ])
    result = generate_grounded_response(
        "How many monitors are available?",
        facts,
        llm,
    )
    assert result.used_fallback is True
    assert result.attempts == 2
    assert result.answer == facts
    assert result.reason == "validation_failed"
    assert llm.calls == 2
    assert get_last_generation().used_fallback is True


def test_generate_accepts_second_attempt():
    facts = "Order 1003: Mouse - Delivered."
    llm = ScriptedLLM([
        "Order 1003 comes with free shipping and store credit.",
        "Order 1003 for the Mouse has status Delivered.",
    ])
    result = generate_grounded_response(
        "What is the status of order 1003?",
        facts,
        llm,
    )
    assert result.used_fallback is False
    assert result.attempts == 2
    assert "Delivered" in result.answer
    assert "1003" in result.answer

def test_ollama_connection_failure_propagates():
    class OfflineLLM:
        def invoke(self, messages):
            raise httpx.ConnectError("Ollama unavailable")

    with pytest.raises(httpx.ConnectError):
        generate_grounded_response(
            "How many monitors are available?",
            "Monitor: 5 units available",
            OfflineLLM()
        )
