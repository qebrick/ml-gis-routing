from pathlib import Path

import numpy as np
import pandas as pd

QUALITY_PATH = Path(
    "data/processed/hmm_interval_quality.parquet"
)
TRAVERSALS_PATH = Path(
    "data/processed/hmm_edge_traversals_pilot.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/ml_interval_training_data.parquet"
)

quality = pd.read_parquet(QUALITY_PATH)
traversals = pd.read_parquet(TRAVERSALS_PATH)

keys = ["trip_id", "point_index"]

# One row per interval in the quality table.
if quality.duplicated(keys).any():
    raise RuntimeError(
        "Quality table contains duplicate interval keys."
    )

# Aggregate the route reconstructed for each GPS interval.
route_summary = (
    traversals.groupby(keys)
    .agg(
        traversal_records=("edge_order", "size"),
        total_traversed_m=("traversed_length_m", "sum"),
        unique_edges=("edge_u", "nunique"),
    )
    .reset_index()
)

df = quality.merge(
    route_summary,
    on=keys,
    how="left",
    validate="one_to_one",
)

df["traversal_records"] = (
    df["traversal_records"]
    .fillna(0)
    .astype(int)
)

df["total_traversed_m"] = (
    df["total_traversed_m"]
    .fillna(0.0)
)

df["unique_edges"] = (
    df["unique_edges"]
    .fillna(0)
    .astype(int)
)

# Compare the two independent representations
# of reconstructed route length.
df["distance_difference_m"] = (
    df["total_traversed_m"] - df["route_distance_m"]
)

print("=== DISTANCE CONSISTENCY ===")
print(
    df["distance_difference_m"]
    .abs()
    .describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    )
    .round(4)
    .to_string()
)

print(
    "Intervals with difference > 1 m:",
    int(
        (df["distance_difference_m"].abs() > 1.0).sum()
    ),
)

flag_columns = [
    col
    for col in df.columns
    if col.startswith("flag_")
]

bad_quality = df[flag_columns].fillna(False).any(axis=1)

valid = (
    ~bad_quality
    & (df["interval_seconds"] == 15)
    & (df["traversal_records"] >= 1)
    & (df["route_distance_m"] > 0)
    & (df["distance_difference_m"].abs() <= 1.0)
)

clean = df.loc[valid].copy()

clean["target_speed_kmh"] = (
    3.6
    * clean["route_distance_m"]
    / clean["interval_seconds"]
)

# This is a diagnostic cap, not a training target clip.
clean = clean.loc[
    clean["target_speed_kmh"].between(1, 130)
].copy()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
clean.to_parquet(OUTPUT_PATH, index=False)

print("\n=== FILTERING ===")
print("All intervals:", len(df))
print("Any quality flag:", int(bad_quality.sum()))
print("Valid before speed filter:", int(valid.sum()))
print("Clean intervals:", len(clean))
print("Trips:", clean["trip_id"].nunique())

print("\n=== CLEAN INTERVAL TYPES ===")
print(
    clean["traversal_records"]
    .value_counts()
    .sort_index()
    .head(15)
    .to_string()
)

print("\n=== TARGET SPEED ===")
print(
    clean["target_speed_kmh"]
    .describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    )
    .round(2)
    .to_string()
)

print("\nSaved:", OUTPUT_PATH)
