from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from backend.app.services.refunds import check_refund_eligibility

class RefundState(TypedDict):
    order_id: int
    eligible: bool
    reason: str
    message: str

def refund_check_node(state: RefundState):
    return check_refund_eligibility(state["order_id"])

def route_refund(state: RefundState) -> str:
    if state["eligible"]:
        return "approved"
    return "rejected"

def approved_node(state: RefundState):
    return {
        "message": "Order is eligible for refund review."
    }


def rejected_node(state: RefundState):
    return {
        "message": f"Refund review unavailable: {state['reason']}"
    }

refund_builder = StateGraph(RefundState)

refund_builder.add_node("check_refund", refund_check_node)
refund_builder.add_node("approved", approved_node)
refund_builder.add_node("rejected", rejected_node)

refund_builder.add_edge(START, "check_refund")

refund_builder.add_conditional_edges(
    "check_refund",
    route_refund,
    {
        "approved": "approved",
        "rejected": "rejected"
    }
)

refund_builder.add_edge("approved", END)
refund_builder.add_edge("rejected", END)

refund_graph = refund_builder.compile()

