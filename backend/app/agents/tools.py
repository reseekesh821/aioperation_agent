from langchain_core.tools import tool
from backend.app.services.inventory import get_inventory, normalize_product_name
from backend.app.services.orders import get_order
from backend.app.agents.refund_graph import refund_graph
from backend.app.services.vector_store import search_policies


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
