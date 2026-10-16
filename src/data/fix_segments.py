from pathlib import Path

import pandas as pd

POINTS_PATH = Path("data/processed/map_matching_clean.parquet")
INTERVALS_PATH = Path("data/processed/gps_intervals.parquet")
OUTPUT_PATH = Path("data/processed/map_matching_segments_v2.parquet")

MIN_SEGMENT_POINTS = 5

points = pd.read_parquet(POINTS_PATH)
intervals = pd.read_parquet(INTERVALS_PATH)

points = points.sort_values(
    ["trip_id", "point_index"]
).reset_index(drop=True)

# Existing segment IDs are kept as boundaries.
points["old_segment_id"] = points["segment_id"]

# Valid interval keys identify a verified transition from i to i+1.
interval_keys = pd.MultiIndex.from_frame(
    intervals[["trip_id", "point_index"]]
)

previous = points.groupby("old_segment_id")["point_index"].shift(1)

# For the current point i, the preceding transition is interval i-1.
preceding_keys = pd.MultiIndex.from_arrays([
    points["trip_id"],
    previous.fillna(-1).astype("int64"),
])

has_preceding_interval = preceding_keys.isin(interval_keys)

first_in_old_segment = previous.isna()
non_consecutive = points["point_index"].ne(previous + 1)

points["new_segment_start"] = (
    first_in_old_segment
    | non_consecutive
    | ~has_preceding_interval
)

points["segment_id"] = points["new_segment_start"].cumsum()

# Remove fragments that are too short.
sizes = points.groupby("segment_id").size()
keep_ids = sizes[sizes >= MIN_SEGMENT_POINTS].index

clean = points[
    points["segment_id"].isin(keep_ids)
].copy()

# Verify every retained adjacent transition has an original interval.
prev_idx = clean.groupby("segment_id")["point_index"].shift(1)
check = pd.MultiIndex.from_arrays([
    clean["trip_id"],
    prev_idx.fillna(-1).astype("int64"),
])
missing_inside = (
    prev_idx.notna()
    & ~check.isin(interval_keys)
)

if missing_inside.any():
    raise RuntimeError(
        "A retained segment contains a missing GPS interval."
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
clean.to_parquet(OUTPUT_PATH, index=False)

print("=== SEGMENTATION V2 ===")
print("Input points:", len(points))
print("Input segments:", points["old_segment_id"].nunique())
print("Missing-interval boundaries added:",
      int((~has_preceding_interval & ~first_in_old_segment).sum()))
print("Output points:", len(clean))
print("Output segments:", clean["segment_id"].nunique())
print("Output trips:", clean["trip_id"].nunique())

print("\nSegment length statistics:")
print(
    clean.groupby("segment_id")
    .size()
    .describe(percentiles=[0.5, 0.9, 0.95])
    .to_string()
)

print("\nSaved:", OUTPUT_PATH)
