from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/processed/map_matching_pilot.parquet")
OUTPUT = Path("data/processed/map_matching_clean.parquet")

MAX_DISTANCE_M = 50.0
MIN_SEGMENT_POINTS = 5

df = pd.read_parquet(INPUT)
df = df.sort_values(["trip_id", "point_index"]).reset_index(drop=True)

print("=== INPUT ===")
print("Trips:", df["trip_id"].nunique())
print("Points:", len(df))

# A point is eligible only if it is sufficiently close to a road edge.
df["is_valid"] = (
    np.isfinite(df["distance_to_edge_m"])
    & (df["distance_to_edge_m"] <= MAX_DISTANCE_M)
)

# A new segment begins after:
# - a change of trip;
# - a missing original GPS index;
# - an invalid point.
#
# Invalid points are retained temporarily to preserve break positions.
new_trip = df["trip_id"].ne(df["trip_id"].shift())
index_gap = df["point_index"].ne(df["point_index"].shift() + 1)
after_invalid = ~df["is_valid"].shift(fill_value=False)

df["segment_start"] = new_trip | index_gap | after_invalid
df["segment_id"] = df["segment_start"].cumsum()

clean = df.loc[df["is_valid"]].copy()

# Retain only sufficiently long continuous segments.
segment_sizes = clean.groupby("segment_id").size()
keep_ids = segment_sizes[
    segment_sizes >= MIN_SEGMENT_POINTS
].index

clean = clean[
    clean["segment_id"].isin(keep_ids)
].copy()

clean = clean.sort_values(
    ["trip_id", "point_index"]
).reset_index(drop=True)

# Check that original GPS indices remain consecutive within segments.
gaps = clean.groupby("segment_id")["point_index"].diff()
bad_gaps = gaps.notna() & gaps.ne(1)

if bad_gaps.any():
    raise RuntimeError(
        "Non-consecutive GPS indices found inside a segment."
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
clean.to_parquet(OUTPUT, index=False)

print("\n=== CLEANED DATA ===")
print("Saved:", OUTPUT)
print("Retained trips:", clean["trip_id"].nunique())
print("Retained segments:", clean["segment_id"].nunique())
print("Retained points:", len(clean))
print("Retention rate:", f"{len(clean) / len(df) * 100:.2f}%")

if not clean.empty:
    print("\nSegment length statistics:")
    print(
        clean.groupby("segment_id")
        .size()
        .describe(percentiles=[0.5, 0.9, 0.95])
        .to_string()
    )

    print("\nDistance to edge statistics (m):")
    print(
        clean["distance_to_edge_m"]
        .describe(percentiles=[0.5, 0.9, 0.95])
        .to_string()
    )

print("\nCleaning completed.")
