import re

from backend.app.services.inventory import (
    find_product_in_text,
    normalize_product_name,
)

# Matches common order-ID phrasings without treating bare numbers as IDs:
#   "order 1003", "order id #1003", "status of order 1003",
#   "order status of 1003", "order status for 1001"
ORDER_ID_PATTERN = re.compile(
    r"\border(?:\s+(?:id|number|#|status)){0,2}\s*(?:(?:of|for)\s+|[:#]\s*)?(\d+)\b",
    re.IGNORECASE,
)

INVENTORY_CUES = (
    "inventory",
    "stock",
    "available",
    "in stock",
    "how many",
    "do you have",
    "any left",
    "units",
)

POLICY_CUES = (
    "refund",
    "policy",
    "warranty",
    "shipping",
    "deadline",
    "store credit",
    "return",
)


def extract_order_id(question: str) -> int | None:
    match = ORDER_ID_PATTERN.search(question)
    if match is None:
        return None
    return int(match.group(1))


def valid_order_reference(question: str, order_id: int) -> bool:
    if not isinstance(order_id, int) or isinstance(order_id, bool):
        return False

    if order_id <= 0:
        return False

    return extract_order_id(question) == order_id


def expected_tool(question: str) -> str | None:
    q = question.casefold()
    order_id = extract_order_id(question)

    if order_id is not None:
        if "policy" in q or "warranty" in q:
            return "policy_lookup"
        if "refund" in q:
            return "refund_lookup"
        return "order_lookup"

    # Policy/refund questions take precedence over inventory cues like "do you have".
    if any(cue in q for cue in POLICY_CUES):
        return "policy_lookup"

    product = find_product_in_text(question)
    if product is not None:
        return "inventory_lookup"

    if any(cue in q for cue in INVENTORY_CUES):
        return "inventory_lookup"

    return None


def validate_tool_call(question: str, call: dict) -> bool:
    name = call.get("name")
    args = call.get("args")
    expected = expected_tool(question)

    if expected is None or name != expected or not isinstance(args, dict):
        return False

    if name in ("order_lookup", "refund_lookup"):
        order_id = args.get("order_id")

        if isinstance(order_id, str) and order_id.isdecimal():
            order_id = int(order_id)

        return valid_order_reference(question, order_id)

    if name == "inventory_lookup":
        product = args.get("product_name")
        if not isinstance(product, str) or not product.strip():
            return False

        canonical = normalize_product_name(product)
        mentioned = find_product_in_text(question)

        # Accept only when the tool argument resolves to a product mentioned
        # in the question (covers singular/plural/alias remapping).
        if canonical is not None and mentioned is not None:
            return canonical == mentioned

        # Unknown catalog names: require the raw argument to appear in the question
        # so the tool can return a clean "not in catalog" result.
        return product.casefold() in question.casefold()

    if name == "policy_lookup":
        return args.get("question") == question

    return False
