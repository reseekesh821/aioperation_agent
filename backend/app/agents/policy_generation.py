from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.app.services.policy_evidence import APPROVED_CLAIMS, NO_POLICY_INFO
from backend.app.agents.response_generation import (
    generate_grounded_response,
    is_response_grounded,
)

# Backwards-compatible alias used by existing tests.
is_answer_grounded = is_response_grounded

POLICY_SYSTEM_PROMPT = """
You are a policy evidence assistant.

Use only the verified policy claims provided.
Select only claim IDs that directly support the answer.

Rules:
- Never invent policies, exceptions, or business conditions.
- Preserve restrictions and eligibility requirements.
- If no claim answers the question, select no claims.
- Never use a claim ID outside the provided evidence.
- Follow the required structured output schema.
"""


class PolicyAnswerPlan(BaseModel):
    claim_ids: list[str] = Field(
        description="Policy IDs supporting the answer"
    )
    style: Literal["direct", "friendly"] = "direct"


def validate_answer_plan(
    plan: PolicyAnswerPlan,
    allowed_ids: set[str],
) -> bool:
    selected = plan.claim_ids

    return (
        len(selected) == len(set(selected))
        and set(selected).issubset(allowed_ids)
    )


def generate_policy_plan(question: str, evidence: list[dict], llm):
    if not evidence:
        return PolicyAnswerPlan(claim_ids=[])

    allowed_ids = {item["id"] for item in evidence}

    context = "\n".join(
        f"{item['id']}: {item['text']}"
        for item in evidence
    )

    model = llm.with_structured_output(
        PolicyAnswerPlan,
        method="json_schema",
    )

    plan = model.invoke([
        SystemMessage(content=POLICY_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {question}\nEvidence:\n{context}"),
    ])

    if not validate_answer_plan(plan, allowed_ids):
        raise ValueError("LLM selected unsupported policy IDs")

    return plan


def render_policy_answer(
    plan: PolicyAnswerPlan,
    evidence: list[dict],
) -> str:
    allowed_ids = {item["id"] for item in evidence}

    if not validate_answer_plan(plan, allowed_ids):
        raise ValueError("Unsupported policy claim")

    if not plan.claim_ids:
        return NO_POLICY_INFO

    return " ".join(
        APPROVED_CLAIMS[claim_id]
        for claim_id in plan.claim_ids
    )


def generate_grounded_policy_answer(question, plan, evidence, llm) -> str:
    """Natural policy answer via the shared grounded generator."""
    verified_facts = render_policy_answer(plan, evidence)
    result = generate_grounded_response(question, verified_facts, llm)
    return result.answer
