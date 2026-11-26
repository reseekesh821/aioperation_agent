import pytest

from backend.app.agents.validation import (
    expected_tool,
    validate_tool_call,
    valid_order_reference,
)


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How many monitors are available?", "inventory_lookup"),
        ("Do you have any mice?", "inventory_lookup"),
        ("What is the status of order 1003?", "order_lookup"),
        ("What is the order status of 1003?", "order_lookup"),
        ("Check order 1001.", "order_lookup"),
        ("What's happening with order 1003?", "order_lookup"),
        ("Is order 1003 eligible for a refund?", "refund_lookup"),
        ("Can I get a refund for order 1001?", "refund_lookup"),
        ("Can I request a refund while shipping?", "policy_lookup"),
        ("What is the refund deadline?", "policy_lookup"),
        ("Does the company offer a lifetime warranty?", "policy_lookup"),
        ("Can I get store credit?", "policy_lookup"),
        ("Hello there", None),
        ("What about 1003?", None),
        (
            "What is the status of order e481f51cbdc54678b7cc49136f2d6af7?",
            "olist_order_lookup",
        ),
        (
            "Is order e481f51cbdc54678b7cc49136f2d6af7 eligible for a refund?",
            "olist_order_lookup",
        ),
        (
            "Look up customer ID 06b8999e2fba1a1fbc88172c00ba8bc7",
            "olist_customer_lookup",
        ),
    ],
)
def test_expected_tool(question, expected):
    assert expected_tool(question) == expected


def test_valid_order_reference_accepts_mentioned_id():
    assert valid_order_reference("status of order 1003", 1003) is True
    assert valid_order_reference("order status of 1003", 1003) is True
    assert valid_order_reference("status of order 1003", 9999) is False
    assert valid_order_reference("just the number 1003", 1003) is False


def test_validate_inventory_allows_canonical_remap():
    question = "How many monitors are available?"
    assert validate_tool_call(
        question,
        {"name": "inventory_lookup", "args": {"product_name": "Monitor"}},
    )


def test_validate_inventory_rejects_unmentioned_product():
    question = "How many monitors are available?"
    assert not validate_tool_call(
        question,
        {"name": "inventory_lookup", "args": {"product_name": "Keyboard"}},
    )


def test_validate_rejects_invented_order_id():
    question = "What is the status of order 1003?"
    assert not validate_tool_call(
        question,
        {"name": "order_lookup", "args": {"order_id": 9999}},
    )


def test_validate_rejects_wrong_tool():
    question = "How many monitors are available?"
    assert not validate_tool_call(
        question,
        {"name": "policy_lookup", "args": {"question": question}},
    )


def test_validate_policy_requires_exact_question():
    question = "What is the refund deadline?"
    assert validate_tool_call(
        question,
        {"name": "policy_lookup", "args": {"question": question}},
    )
    assert not validate_tool_call(
        question,
        {"name": "policy_lookup", "args": {"question": "other"}},
    )
