from pathlib import Path
import csv

data_dir = Path(__file__).resolve().parents[1] / "data/raw/olist"

for file in sorted(data_dir.glob("*.csv")):
    with file.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        columns = next(reader)
        row_count = sum(1 for row in reader)

    print(f"\n{file.name}")
    print(f"Rows: {row_count:,}")
    print(f"Columns: {', '.join(columns)}")
