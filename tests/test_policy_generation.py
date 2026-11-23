import pytest

from backend.app.agents.policy_generation import (
    PolicyAnswerPlan,
    render_policy_answer,
    is_answer_grounded,
)
from backend.app.services.policy_evidence import APPROVED_CLAIMS, NO_POLICY_INFO


def test_unsupported_claim_rejected():
    plan = PolicyAnswerPlan(claim_ids=["refund_99"])
    evidence = [{"id": "refund_3", "text": "Shipping restriction"}]

    with pytest.raises(ValueError, match="Unsupported policy claim"):
        render_policy_answer(plan, evidence)


def test_empty_plan_returns_no_info():
    plan = PolicyAnswerPlan(claim_ids=[])
    evidence = [{"id": "refund_3", "text": "Shipping restriction"}]
    assert render_policy_answer(plan, evidence) == NO_POLICY_INFO


def test_grounding_rejects_warranty_hallucination():
    facts = APPROVED_CLAIMS["refund_4"]
    answer = (
        "Damaged products are covered by our lifetime warranty "
        "and can be replaced immediately."
    )
    assert not is_answer_grounded(answer, facts)
