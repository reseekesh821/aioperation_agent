import os
import re
import psycopg

# Translate user language into canonical product names.
PRODUCT_ALIASES: dict[str, str] = {
    "keyboard": "Keyboard",
    "keyboards": "Keyboard",
    "mouse": "Mouse",
    "mice": "Mouse",
    "monitor": "Monitor",
    "monitors": "Monitor",
    "headphone": "Headphones",
    "headphones": "Headphones",
}


def list_products() -> list[str]:
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        rows = conn.execute(
            "SELECT product_name FROM inventory"
        ).fetchall()

    return [row[0] for row in rows]


def normalize_product_name(product_name: str) -> str | None:
    if not isinstance(product_name, str):
        return None

    key = product_name.strip().casefold()
    if not key:
        return None

    if key in PRODUCT_ALIASES:
        return PRODUCT_ALIASES[key]

    catalog = {
        name.casefold(): name
        for name in list_products()
    }

    if key in catalog:
        return catalog[key]

    if key.endswith("es") and key[:-2] in catalog:
        return catalog[key[:-2]]

    if key.endswith("s") and key[:-1] in catalog:
        return catalog[key[:-1]]

    return None


def find_product_in_text(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None

    aliases = sorted(PRODUCT_ALIASES, key=len, reverse=True)
    lowered = text.casefold()

    for alias in aliases:
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return PRODUCT_ALIASES[alias]

    for name in list_products():
        if re.search(rf"\b{re.escape(name.casefold())}\b", lowered):
            return name

    return None


def get_inventory(product_name: str) -> int | None:
    canonical = normalize_product_name(product_name)

    if canonical is None:
        return None

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        row = conn.execute(
            "SELECT quantity FROM inventory WHERE product_name = %s",
            (canonical,)
        ).fetchone()

    return None if row is None else row[0]
