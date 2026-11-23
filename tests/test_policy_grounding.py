import pytest

from backend.app.agents.agent_graph import agent_graph
from backend.app.agents.response_generation import get_last_generation
from backend.app.services.policy_evidence import NO_POLICY_INFO


def _has_any(text: str, options: list[str]) -> bool:
    lowered = text.lower()
    return any(option.lower() in lowered for option in options)


@pytest.mark.parametrize(
    "question, required_any, forbidden_terms",
    [
        (
            "Can I request a refund while shipping?",
            [["shipping"], ["standard"]],
            ["store credit", "lifetime warranty"],
        ),
        (
            "Does the company offer a lifetime warranty?",
            [[
                "do not specify",
                "don't specify",
                "does not specify",
                "do not mention",
                "don't mention",
                "does not mention",
                "not specified",
            ]],
            ["lifetime warranty is offered", "covered by warranty"],
        ),
        (
            "What is the refund deadline?",
            [["30 days"], ["delivery"]],
            ["store credit", "lifetime warranty", "90 days"],
        ),
        (
            "Can I get store credit?",
            [[
                "do not specify",
                "don't specify",
                "does not specify",
                "do not mention",
                "don't mention",
                "not specified",
            ]],
            ["we offer store credit", "yes, store credit"],
        ),
    ],
)
def test_policy_grounding(question, required_any, forbidden_terms):
    result = agent_graph.invoke({
        "messages": [("user", question)]
    })

    answer = result["messages"][-1].content
    tools_used = [
        m.name for m in result["messages"]
        if m.type == "tool"
    ]

    assert tools_used == ["policy_lookup"]
    assert get_last_generation() is not None

    for options in required_any:
        assert _has_any(answer, options), f"missing one of {options!r} in {answer!r}"

    lowered = answer.lower()
    for term in forbidden_terms:
        assert term.lower() not in lowered, f"found forbidden {term!r} in {answer!r}"


def test_warranty_uses_uncertainty_shape():
    result = agent_graph.invoke({
        "messages": [("user", "Does the company offer a lifetime warranty?")]
    })
    answer = result["messages"][-1].content
    assert _has_any(
        answer,
        [
            "do not specify",
            "don't specify",
            "does not specify",
            "do not mention",
            "don't mention",
            "not specified",
            NO_POLICY_INFO,
        ],
    )
