from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "olist"

TABLES = [
    ("olist.customers", "olist_customers_dataset.csv", 99441),
    ("olist.orders", "olist_orders_dataset.csv", 99441),
    ("olist.products", "olist_products_dataset.csv", 32951),
    ("olist.sellers", "olist_sellers_dataset.csv", 3095),
    ("olist.order_items", "olist_order_items_dataset.csv", 112650),
    ("olist.payments", "olist_order_payments_dataset.csv", 103886),
    ("olist.reviews", "olist_order_reviews_dataset.csv", 99224),
]

def run_sql(sql):
    result = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "db", "sh", "-c",
            f'psql -U "${{POSTGRES_USER:-postgres}}" '
            f'-d "${{POSTGRES_DB:-postgres}}" -Atqc "{sql}"'
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()

for table, filename, expected in TABLES:
    count = int(run_sql(f"SELECT COUNT(*) FROM {table}"))

    if count == expected:
        print(f"{table}: already loaded, skipping")
        continue

    if count != 0:
        raise RuntimeError(
            f"{table} has {count} rows; expected 0 or {expected}. "
            "Import stopped to protect existing data."
        )

    print(f"Importing {filename}...")

    with (DATA / filename).open("rb") as file:
        subprocess.run(
            [
                "docker", "compose", "exec", "-T", "db", "sh", "-c",
                f'psql -U "${{POSTGRES_USER:-postgres}}" '
                f'-d "${{POSTGRES_DB:-postgres}}" '
                f'-v ON_ERROR_STOP=1 '
                f'-c "COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true)"'
            ],
            stdin=file,
            cwd=ROOT,
            check=True,
        )

    print(f"{table}: {run_sql(f'SELECT COUNT(*) FROM {table}')} rows")
