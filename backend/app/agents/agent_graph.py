import logging
from uuid import uuid4

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import ToolMessage, AIMessage

from backend.app.agents.ai_errors import is_ai_service_error
from backend.app.agents.validation import (
    expected_tool,
    extract_order_id,
    extract_olist_order_id,
    extract_olist_customer_id,
    validate_tool_call,
)
from backend.app.services.inventory import find_product_in_text
from backend.app.services.policy_evidence import select_policy_evidence
from backend.app.agents.policy_generation import (
    generate_policy_plan,
    render_policy_answer,
)
from backend.app.agents.response_generation import (
    generate_grounded_response,
    generate_olist_response,
)
from backend.app.agents.tools import (
    inventory_lookup,
    order_lookup,
    refund_lookup,
    policy_lookup,
    olist_order_lookup,
    olist_customer_lookup,
)
from backend.app.services.vector_store import ensure_policies_indexed

logger = logging.getLogger(__name__)

# Ensure the Chroma collection has policy embeddings before serving traffic.
ensure_policies_indexed()

tools = [
    inventory_lookup,
    order_lookup,
    refund_lookup,
    policy_lookup,
    olist_order_lookup,
    olist_customer_lookup,
]

llm = ChatOllama(model="llama3.1:8b", temperature=0)
# Do not convert tool/runtime failures into ToolMessage success payloads.
tool_node = ToolNode(tools, handle_tool_errors=False)

CLARIFY_MESSAGE = (
    "I can help with inventory, order status, refund eligibility, "
    "refund policy questions, or historical Olist order/customer lookups. "
    "Please include a product name, numeric order ID, or 32-character Olist "
    "order or customer ID."
)
INVALID_TOOL_MESSAGE = "I couldn't validate the tool call."
POLICY_FALLBACK = "The available policies do not specify this information."
MISSING_PRODUCT_MESSAGE = (
    "Please specify which product you want inventory for."
)


def _latest_human_text(messages) -> str:
    return next(
        m.content for m in reversed(messages)
        if m.type == "human"
    )


def _forced_tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": name,
            "args": args,
            "id": str(uuid4()),
        }],
    )


def _build_tool_response(question: str, expected: str):
    """Prefer deterministic args; fall back to clarify when needed."""
    if expected == "policy_lookup":
        return _forced_tool_call(
            "policy_lookup",
            {"question": question},
        )

    if expected == "inventory_lookup":
        product = find_product_in_text(question)
        if product is None:
            return AIMessage(content=MISSING_PRODUCT_MESSAGE)
        return _forced_tool_call(
            "inventory_lookup",
            {"product_name": product},
        )

    if expected in ("order_lookup", "refund_lookup"):
        order_id = extract_order_id(question)
        if order_id is None:
            return AIMessage(content=INVALID_TOOL_MESSAGE)
        return _forced_tool_call(
            expected,
            {"order_id": order_id},
        )

    if expected == "olist_order_lookup":
        olist_order_id = extract_olist_order_id(question)
        if olist_order_id is None:
            return AIMessage(content=INVALID_TOOL_MESSAGE)
        return _forced_tool_call(
            "olist_order_lookup",
            {"order_id": olist_order_id},
        )

    if expected == "olist_customer_lookup":
        olist_customer_id = extract_olist_customer_id(question)
        if olist_customer_id is None:
            return AIMessage(content=INVALID_TOOL_MESSAGE)
        return _forced_tool_call(
            "olist_customer_lookup",
            {"customer_id": olist_customer_id},
        )

    return AIMessage(content=CLARIFY_MESSAGE)


def _final_answer_from_tool(question: str, tool_message: ToolMessage) -> str:
    """Build verified facts, then generate a grounded natural answer."""
    if tool_message.name == "policy_lookup":
        chunks = [
            line for line in tool_message.content.splitlines()
            if line.strip()
        ]
        try:
            evidence = select_policy_evidence(question, chunks)
            plan = generate_policy_plan(question, evidence, llm)
            verified_facts = render_policy_answer(plan, evidence)
        except Exception as exc:
            if is_ai_service_error(exc):
                raise
            logger.exception("policy planning failed; using safe fallback")
            return POLICY_FALLBACK

        result = generate_grounded_response(question, verified_facts, llm)
        return result.answer

    if tool_message.name in ("olist_order_lookup", "olist_customer_lookup"):
        result = generate_olist_response(question, tool_message.content, llm)
        return result.answer

    verified_facts = tool_message.content
    result = generate_grounded_response(question, verified_facts, llm)
    return result.answer


def assistant_node(state: MessagesState):
    messages = state["messages"]

    if isinstance(messages[-1], ToolMessage):
        question = _latest_human_text(messages)
        answer = _final_answer_from_tool(question, messages[-1])
        return {"messages": [AIMessage(content=answer)]}

    question = _latest_human_text(messages)
    expected = expected_tool(question)

    if expected is None:
        return {"messages": [AIMessage(content=CLARIFY_MESSAGE)]}

    response = _build_tool_response(question, expected)

    # Clarifying replies have no tool calls.
    if not getattr(response, "tool_calls", None):
        return {"messages": [response]}

    if (
        len(response.tool_calls) != 1
        or not validate_tool_call(question, response.tool_calls[0])
    ):
        return {"messages": [AIMessage(content=INVALID_TOOL_MESSAGE)]}

    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

agent_graph = builder.compile()
