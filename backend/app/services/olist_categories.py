"""Minimal product-category translation for historical Olist labels."""

from __future__ import annotations

import csv
from pathlib import Path

_TRANSLATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "raw"
    / "olist"
    / "product_category_name_translation.csv"
)

_TRANSLATIONS: dict[str, str] | None = None


def load_category_translations() -> dict[str, str]:
    global _TRANSLATIONS
    if _TRANSLATIONS is not None:
        return _TRANSLATIONS

    translations: dict[str, str] = {}
    if _TRANSLATION_PATH.is_file():
        # utf-8-sig strips the BOM present in the Olist translation CSV.
        with _TRANSLATION_PATH.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                portuguese = (row.get("product_category_name") or "").strip()
                english = (row.get("product_category_name_english") or "").strip()
                if portuguese and english:
                    translations[portuguese] = english

    _TRANSLATIONS = translations
    return _TRANSLATIONS


def describe_product_category(category_name: str | None) -> str:
    """Describe an Olist category label without presenting it as a product name."""
    if not category_name:
        return "product category=unknown (category label, not a specific product name)"

    translations = load_category_translations()
    english = translations.get(category_name)
    if english:
        return (
            f"product category={english} "
            f"(Olist category label '{category_name}'; not a specific product name)"
        )

    return (
        f"product category={category_name} "
        "(category label, not a specific product name)"
    )
