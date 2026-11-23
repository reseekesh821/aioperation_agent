from datetime import date, timedelta

from backend.app.services.refunds import check_refund_eligibility


def test_order_not_found():
    result = check_refund_eligibility(9999)

    assert result["eligible"] is False
    assert result["reason"] == "Order not found"


def test_expired_refund(monkeypatch):
    fake_order = {
        "status": "Delivered",
        "delivered_at": (date.today() - timedelta(days=45)).isoformat(),
    }

    monkeypatch.setattr(
        "backend.app.services.refunds.get_order",
        lambda order_id: fake_order,
    )

    result = check_refund_eligibility(1003)

    assert result["eligible"] is False
    assert result["reason"] == "Refund deadline expired"


def test_eligible_refund(monkeypatch):
    fake_order = {
        "status": "Delivered",
        "delivered_at": (date.today() - timedelta(days=7)).isoformat(),
    }

    monkeypatch.setattr(
        "backend.app.services.refunds.get_order",
        lambda order_id: fake_order,
    )

    result = check_refund_eligibility(1003)

    assert result["eligible"] is True
    assert result["reason"] == "Eligible for refund review"


def test_undelivered_refund(monkeypatch):
    fake_order = {"status": "Shipped"}

    monkeypatch.setattr(
        "backend.app.services.refunds.get_order",
        lambda order_id: fake_order,
    )

    result = check_refund_eligibility(1001)

    assert result["eligible"] is False
    assert result["reason"] == "Order is not delivered"


def test_missing_delivery_date(monkeypatch):
    monkeypatch.setattr(
        "backend.app.services.refunds.get_order",
        lambda order_id: {"status": "Delivered"},
    )

    result = check_refund_eligibility(1003)
    assert result["eligible"] is False
    assert result["reason"] == "Delivery date unavailable"
