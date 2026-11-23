import pytest

from backend.app.services.inventory import (
    get_inventory,
    normalize_product_name,
    find_product_in_text,
)
from backend.app.agents.tools import inventory_lookup


@pytest.mark.parametrize(
    "raw, canonical",
    [
        ("Monitor", "Monitor"),
        ("monitor", "Monitor"),
        ("monitors", "Monitor"),
        ("Keyboard", "Keyboard"),
        ("keyboards", "Keyboard"),
        ("Mouse", "Mouse"),
        ("mice", "Mouse"),
        ("headphones", "Headphones"),
        ("headphone", "Headphones"),
    ],
)
def test_normalize_product_name(raw, canonical):
    assert normalize_product_name(raw) == canonical


def test_normalize_unknown_product():
    assert normalize_product_name("laptop") is None
    assert normalize_product_name("") is None


@pytest.mark.parametrize(
    "raw, quantity",
    [
        ("Monitor", 5),
        ("monitors", 5),
        ("Keyboard", 12),
        ("mice", 0),
        ("Headphones", 8),
    ],
)
def test_get_inventory_quantities(raw, quantity):
    assert get_inventory(raw) == quantity


def test_get_inventory_unknown():
    assert get_inventory("tablet") is None


def test_zero_stock_distinct_from_missing():
    assert get_inventory("Mouse") == 0
    assert get_inventory("Unicorn") is None


def test_find_product_in_text():
    assert find_product_in_text("How many monitors are available?") == "Monitor"
    assert find_product_in_text("Do you have any mice?") == "Mouse"
    assert find_product_in_text("What about shipping?") is None


def test_inventory_lookup_tool_messages():
    assert "5 units available" in inventory_lookup.invoke({"product_name": "monitors"})
    assert "out of stock" in inventory_lookup.invoke({"product_name": "mice"})
    assert "not in the catalog" in inventory_lookup.invoke({"product_name": "tablet"})
