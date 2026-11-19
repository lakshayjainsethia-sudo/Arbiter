
from pathlib import Path
import pandas as pd

# Path to the downloaded LendingClub dataset
DATA_PATH = Path(
    "data/raw/lendingclub/LC_loans_granting_model_dataset.csv"
)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

print("Loading LendingClub dataset...")

# Load only 5,000 rows for initial inspection
df = pd.read_csv(DATA_PATH, nrows=5000, low_memory=False)

print("\n--- Dataset dimensions (sample) ---")
print(df.shape)

print("\n--- Column names ---")
for column in df.columns:
    print(column)

print("\n--- First five rows ---")
print(df.head())

print("\n--- Missing values: top 15 columns ---")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\n--- Possible target columns ---")
for column in df.columns:
    if any(
        word in column.lower()
        for word in ["default", "status", "target", "class"]
    ):
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False).head(10))

print("\nInspection complete.")
