from datetime import date
from backend.app.services.orders import get_order

def check_refund_eligibility(order_id: int):
    order = get_order(order_id)

    if order is None:
        return {"eligible": False, "reason": "Order not found"}

    if order["status"] != "Delivered":
        return {"eligible": False, "reason": "Order is not delivered"}

    delivered_at = order.get("delivered_at")

    if not isinstance(delivered_at, str):
        return {"eligible": False, "reason": "Delivery date unavailable"}

    try:
        delivery_date = date.fromisoformat(delivered_at)
    except ValueError:
        return {"eligible": False, "reason": "Invalid delivery date"}

    days = (date.today() - delivery_date).days

    if days < 0:
        return {"eligible": False, "reason": "Invalid future delivery date"}

    if days > 30:
        return {"eligible": False, "reason": "Refund deadline expired"}

    return {"eligible": True, "reason": "Eligible for refund review"}