import os
import psycopg
from psycopg.rows import dict_row


def get_order(order_id: int):
    with psycopg.connect(
        os.environ["DATABASE_URL"],
        row_factory=dict_row
    ) as conn:

        order = conn.execute(
            "SELECT * FROM orders WHERE order_id = %s",
            (order_id,)
        ).fetchone()

    if order and order["delivered_at"]:
        order["delivered_at"] = order["delivered_at"].isoformat()

    return order