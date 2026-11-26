import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def get_olist_order(order_id: str):
    with psycopg.connect(
        os.environ["DATABASE_URL"],
        row_factory=dict_row
    ) as conn:

        order = conn.execute(
            """
            SELECT
                order_id,
                order_status,
                order_purchase_timestamp,
                order_delivered_customer_date,
                order_estimated_delivery_date
            FROM olist.orders
            WHERE order_id = %s
            """,
            (order_id,)
        ).fetchone()

        if order is None:
            return None

        items = conn.execute(
            """
            SELECT
                oi.order_item_id,
                oi.product_id,
                p.product_category_name,
                oi.price,
                oi.freight_value
            FROM olist.order_items oi
            JOIN olist.products p
                ON oi.product_id = p.product_id
            WHERE oi.order_id = %s
            ORDER BY oi.order_item_id
            """,
            (order_id,)
        ).fetchall()

        order["items"] = items
        order["source"] = "olist_historical"


        payments = conn.execute(
            """
            SELECT payment_type, payment_installments, payment_value
            FROM olist.payments
            WHERE order_id = %s
            ORDER BY payment_sequential
            """,
            (order_id,)
        ).fetchall()

        reviews = conn.execute(
            """
            SELECT review_score, review_comment_title,
                   review_comment_message
            FROM olist.reviews
            WHERE order_id = %s
            """,
            (order_id,)
        ).fetchall()

        order["payments"] = payments
        order["reviews"] = reviews


        return order
