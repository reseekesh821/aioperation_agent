from pathlib import Path
import kagglehub

# Locate our existing project directory
project_root = Path(__file__).resolve().parents[1]

# Where the downloaded dataset will be stored
destination = project_root / "data" / "raw" / "olist"

# Download the Olist ecommerce dataset
path = kagglehub.dataset_download(
    "olistbr/brazilian-ecommerce",
    output_dir=str(destination)
)

print("Dataset downloaded to:", path)

# Display downloaded CSV files
for file in sorted(destination.rglob("*.csv")):
    print(file.name)
