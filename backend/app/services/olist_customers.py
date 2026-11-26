import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def get_olist_customer(customer_id: str):
    """Return historical Olist customer fields and matching order IDs/statuses."""
    with psycopg.connect(
        os.environ["DATABASE_URL"],
        row_factory=dict_row,
    ) as conn:
        customer = conn.execute(
            """
            SELECT
                customer_id,
                customer_city,
                customer_state
            FROM olist.customers
            WHERE customer_id = %s
            """,
            (customer_id,),
        ).fetchone()

        if customer is None:
            return None

        orders = conn.execute(
            """
            SELECT
                order_id,
                order_status,
                order_purchase_timestamp
            FROM olist.orders
            WHERE customer_id = %s
            ORDER BY order_purchase_timestamp NULLS LAST, order_id
            """,
            (customer_id,),
        ).fetchall()

        customer["orders"] = orders
        customer["source"] = "olist_historical"
        return customer
