from backend.app.services.orders import get_order
from backend.app.agents.tools import order_lookup, refund_lookup


def test_get_order_existing():
    order = get_order(1003)
    assert order is not None
    assert order["status"] == "Delivered"
    assert order["product"] == "Mouse"


def test_get_order_missing():
    assert get_order(9999) is None


def test_order_lookup_tool():
    result = order_lookup.invoke({"order_id": 1003})
    assert "1003" in result
    assert "Delivered" in result

    missing = order_lookup.invoke({"order_id": 9999})
    assert "not found" in missing.lower()


def test_refund_lookup_tool_for_undelivered():
    result = refund_lookup.invoke({"order_id": 1001})
    assert "1001" in result
    assert "unavailable" in result.lower() or "not" in result.lower()
