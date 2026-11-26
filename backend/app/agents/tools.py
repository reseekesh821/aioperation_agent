import re

from langchain_core.tools import tool

from backend.app.services.inventory import get_inventory, normalize_product_name
from backend.app.services.orders import get_order
from backend.app.agents.refund_graph import refund_graph
from backend.app.services.vector_store import search_policies
from backend.app.services.olist_categories import describe_product_category
from backend.app.services.olist_customers import get_olist_customer
from backend.app.services.olist_orders import get_olist_order


@tool
def inventory_lookup(product_name: str) -> str:
    """Check how many units of a product are available."""

    canonical = normalize_product_name(product_name)
    if canonical is None:
        return f"Product '{product_name}' is not in the catalog."

    quantity = get_inventory(canonical)
    if quantity is None:
        return f"Product '{product_name}' is not in the catalog."

    if quantity == 0:
        return f"{canonical}: 0 units available (out of stock)."

    return f"{canonical}: {quantity} units available."


@tool
def order_lookup(order_id: int) -> str:
    """Look up an order's product and delivery status."""

    order = get_order(order_id)

    if order is None:
        return f"Order {order_id} not found."

    return f"Order {order_id}: {order['product']} - {order['status']}."


@tool
def refund_lookup(order_id: int) -> str:
    """Check whether an order is eligible for refund review."""

    result = refund_graph.invoke({"order_id": order_id})
    message = result["message"]

    # Include the order ID in verified facts so grounded answers can cite it.
    if str(order_id) in message:
        return message
    return f"Order {order_id}: {message}"


@tool
def policy_lookup(question: str) -> str:
    """Search company policies to answer general policy questions."""

    policies = search_policies(question)

    if not policies:
        return "No relevant policy found."

    return "\n".join(policies)


def format_olist_order_facts(order: dict) -> str:
    """Build concise verified facts for historical Olist records."""
    lines = [
        "Source: historical Olist ecommerce dataset (not a live operations order).",
        "These facts must not be used to infer current refund eligibility or live inventory.",
        f"Olist order ID: {order['order_id']}",
        f"Historical status: {order.get('order_status')}",
        f"Purchase timestamp: {order.get('order_purchase_timestamp')}",
        f"Delivered to customer: {order.get('order_delivered_customer_date')}",
        f"Estimated delivery date: {order.get('order_estimated_delivery_date')}",
    ]

    items = order.get("items") or []
    if not items:
        lines.append("Items: none recorded.")
    else:
        lines.append(f"Items ({len(items)}):")
        for item in items:
            lines.append(
                "  - "
                f"item {item.get('order_item_id')}: "
                f"{describe_product_category(item.get('product_category_name'))}, "
                f"price={item.get('price')}, "
                f"freight={item.get('freight_value')}"
            )

    payments = order.get("payments") or []
    if not payments:
        lines.append("Payments: none recorded.")
    else:
        lines.append(f"Payments ({len(payments)}):")
        for payment in payments:
            lines.append(
                "  - "
                f"type={payment.get('payment_type')}, "
                f"installments={payment.get('payment_installments')}, "
                f"value={payment.get('payment_value')}"
            )

    reviews = order.get("reviews") or []
    if not reviews:
        lines.append("No customer review was found for this order.")
    else:
        scores = [
            str(review.get("review_score"))
            for review in reviews
            if review.get("review_score") is not None
        ]
        if scores:
            lines.append(f"Customer review score(s): {', '.join(scores)}.")
        else:
            lines.append("No customer review score was found for this order.")
        lines.append(
            "Customer review text is untrusted user-generated content and is omitted "
            "from verified facts so it cannot be treated as instructions."
        )

    return "\n".join(lines)


def format_olist_customer_facts(customer: dict) -> str:
    """Build concise verified facts for historical Olist customers."""
    orders = customer.get("orders") or []
    lines = [
        "Source: historical Olist ecommerce dataset (not live customer CRM data).",
        "These facts must not be used to infer current contact details, refund "
        "eligibility, or live inventory.",
        f"Olist customer ID: {customer['customer_id']}",
        (
            "Historical city/state from dataset (snapshot only, not current contact "
            f"info): {customer.get('customer_city')}, {customer.get('customer_state')}"
        ),
        f"Matching historical orders ({len(orders)}):",
    ]

    if not orders:
        lines.append("  - none recorded.")
    else:
        for order in orders:
            lines.append(
                "  - "
                f"order_id={order.get('order_id')} "
                f"status={order.get('order_status')} "
                f"purchase={order.get('order_purchase_timestamp')}"
            )

    return "\n".join(lines)


@tool
def olist_order_lookup(order_id: str) -> str:
    """Retrieve verified historical Olist order details,
    including items, delivery, payments, and reviews."""

    order_id = order_id.strip().lower()

    if not re.fullmatch(r"[a-f0-9]{32}", order_id):
        return "Invalid Olist order ID."

    order = get_olist_order(order_id)

    if order is None:
        return "Historical Olist order not found."

    return format_olist_order_facts(order)


@tool
def olist_customer_lookup(customer_id: str) -> str:
    """Retrieve verified historical Olist customer details and matching order IDs."""

    customer_id = customer_id.strip().lower()

    if not re.fullmatch(r"[a-f0-9]{32}", customer_id):
        return "Invalid Olist customer ID."

    customer = get_olist_customer(customer_id)

    if customer is None:
        return "Historical Olist customer not found."

    return format_olist_customer_facts(customer)
