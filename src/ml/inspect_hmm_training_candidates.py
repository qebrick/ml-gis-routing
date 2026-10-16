from pathlib import Path

import pandas as pd

QUALITY_PATH = Path(
    "data/processed/hmm_interval_quality.parquet"
)
TRAVERSALS_PATH = Path(
    "data/processed/hmm_edge_traversals_pilot.parquet"
)

quality = pd.read_parquet(QUALITY_PATH)
traversals = pd.read_parquet(TRAVERSALS_PATH)

print("=== INPUT SCHEMAS ===")
print("Quality rows:", len(quality))
print("Quality columns:", list(quality.columns))
print("Traversal rows:", len(traversals))
print("Traversal columns:", list(traversals.columns))

keys = ["trip_id", "point_index"]

for name, df in [
    ("quality", quality),
    ("traversals", traversals),
]:
    missing = set(keys) - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{name}: missing keys {sorted(missing)}"
        )

counts = (
    traversals.groupby(keys)
    .size()
    .rename("traversal_records")
    .reset_index()
)

summary = quality.merge(
    counts,
    on=keys,
    how="left",
    validate="one_to_one",
)

summary["traversal_records"] = (
    summary["traversal_records"]
    .fillna(0)
    .astype(int)
)

print("\n=== TRAVERSALS PER INTERVAL ===")
print(
    summary["traversal_records"]
    .describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
    )
    .round(2)
    .to_string()
)

print("\n=== INTERVAL COUNTS ===")
print("All intervals:", len(summary))
print(
    "Single traversal:",
    int((summary["traversal_records"] == 1).sum()),
)
print(
    "Multiple traversals:",
    int((summary["traversal_records"] > 1).sum()),
)
print(
    "No traversals:",
    int((summary["traversal_records"] == 0).sum()),
)

flag_columns = [
    col
    for col in quality.columns
    if col.startswith("flag_")
]

print("\n=== QUALITY FLAGS ===")
for col in flag_columns:
    print(col, int(quality[col].fillna(False).sum()))

print("\n=== TRAVERSAL COUNT DISTRIBUTION ===")
print(
    summary["traversal_records"]
    .value_counts()
    .sort_index()
    .head(20)
    .to_string()
)
