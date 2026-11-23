from backend.app.services.policy_evidence import (
    select_policy_evidence,
    is_unsupported_policy_topic,
    APPROVED_CLAIMS,
)
from backend.app.agents.policy_generation import (
    PolicyAnswerPlan,
    render_policy_answer,
    is_answer_grounded,
    generate_grounded_policy_answer,
)
from backend.app.agents.response_generation import get_last_generation


SHIPPING_CHUNKS = [
    "3. Orders still processing or shipping cannot submit a standard refund request and are not eligible for standard refund review.",
    "1. Standard refund requests can be submitted only after delivery and within 30 days of the delivery date.",
]

DEADLINE_CHUNKS = [
    "1. Standard refund requests can be submitted only after delivery and within 30 days of the delivery date.",
    "2. Only delivered orders are eligible for standard refund review.",
]


def test_unsupported_topics_detected():
    assert is_unsupported_policy_topic("lifetime warranty")
    assert is_unsupported_policy_topic("Can I get store credit?")
    assert not is_unsupported_policy_topic("What is the refund deadline?")


def test_select_evidence_drops_unsupported_topics():
    evidence = select_policy_evidence(
        "Does the company offer a lifetime warranty?",
        SHIPPING_CHUNKS,
    )
    assert evidence == []


def test_select_evidence_keeps_signal_matches():
    evidence = select_policy_evidence(
        "Can I request a refund while shipping?",
        SHIPPING_CHUNKS,
    )
    assert any(item["id"] == "refund_3" for item in evidence)


def test_select_evidence_keeps_retrieved_without_exact_keywords():
    evidence = select_policy_evidence(
        "Tell me about refund timing after I receive my package.",
        DEADLINE_CHUNKS,
    )
    assert evidence
    assert {item["id"] for item in evidence} <= {"refund_1", "refund_2"}


def test_render_uses_approved_claims_only():
    plan = PolicyAnswerPlan(claim_ids=["refund_1"])
    evidence = [{"id": "refund_1", "text": "deadline rule"}]
    assert render_policy_answer(plan, evidence) == APPROVED_CLAIMS["refund_1"]


def test_is_answer_grounded_accepts_paraphrase():
    facts = APPROVED_CLAIMS["refund_3"]
    answer = (
        "Orders that are still shipping or processing cannot submit "
        "a standard refund request."
    )
    assert is_answer_grounded(answer, facts)


def test_is_answer_grounded_rejects_store_credit():
    facts = APPROVED_CLAIMS["refund_1"]
    answer = "You can get store credit within 30 days after delivery."
    assert not is_answer_grounded(answer, facts)


def test_is_answer_grounded_rejects_invented_number():
    facts = APPROVED_CLAIMS["refund_1"]
    answer = (
        "Standard refund requests must be submitted within 90 days "
        "after delivery."
    )
    assert not is_answer_grounded(answer, facts)


def test_grounded_generation_falls_back_on_bad_candidate():
    plan = PolicyAnswerPlan(claim_ids=["refund_1"])
    evidence = [{"id": "refund_1", "text": "deadline"}]

    class FakeLLM:
        def invoke(self, messages):
            class Resp:
                content = "We offer store credit and a lifetime warranty."

            return Resp()

    answer = generate_grounded_policy_answer(
        "What is the refund deadline?",
        plan,
        evidence,
        FakeLLM(),
    )
    assert answer == APPROVED_CLAIMS["refund_1"]
    meta = get_last_generation()
    assert meta is not None
    assert meta.used_fallback is True
    assert meta.attempts == 2
