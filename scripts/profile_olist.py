import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data/raw/olist"


def get_ids(filename, column):
    ids = set()
    duplicates = 0
    missing = 0

    with (DATA_DIR / filename).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            value = row[column].strip()

            if not value:
                missing += 1
            elif value in ids:
                duplicates += 1
            else:
                ids.add(value)

    print(f"{filename}: {len(ids):,} unique IDs")
    print(f"  Missing IDs: {missing:,}")
    print(f"  Duplicate IDs: {duplicates:,}")
    return ids


def check_references(filename, column, valid_ids):
    missing = 0

    with (DATA_DIR / filename).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row[column].strip() not in valid_ids:
                missing += 1

    print(f"{filename} -> {column}: {missing:,} invalid references")


customers = get_ids("olist_customers_dataset.csv", "customer_id")
orders = get_ids("olist_orders_dataset.csv", "order_id")
products = get_ids("olist_products_dataset.csv", "product_id")
sellers = get_ids("olist_sellers_dataset.csv", "seller_id")

print("\nChecking relationships...")

check_references("olist_orders_dataset.csv", "customer_id", customers)
check_references("olist_order_items_dataset.csv", "order_id", orders)
check_references("olist_order_items_dataset.csv", "product_id", products)
check_references("olist_order_items_dataset.csv", "seller_id", sellers)
check_references("olist_order_payments_dataset.csv", "order_id", orders)
check_references("olist_order_reviews_dataset.csv", "order_id", orders)
