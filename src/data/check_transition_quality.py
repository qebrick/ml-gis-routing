from pathlib import Path

import numpy as np
import pandas as pd

TRANSITIONS = Path("data/processed/edge_transition_diagnostics.parquet")
INTERVALS = Path("data/processed/gps_intervals.parquet")
OUTPUT = Path("data/processed/transition_quality.parquet")

trans = pd.read_parquet(TRANSITIONS)
gps = pd.read_parquet(INTERVALS)

gps = gps[
    [
        "trip_id",
        "point_index",
        "distance_m",
        "interval_seconds",
        "observed_speed_kmh",
    ]
].rename(columns={"point_index": "point_index_1"})

merged = trans.merge(
    gps,
    on=["trip_id", "point_index_1"],
    how="left",
    validate="many_to_one",
    indicator=True,
)

missing = merged["_merge"] != "both"

print("=== MERGE QUALITY ===")
print("Original transitions:", len(merged))
print("Transitions without GPS interval:", int(missing.sum()))

# Exclude transitions without a verified original interval.
valid = merged.loc[~missing].copy()

# Ensure that interval durations are physically meaningful.
valid = valid[
    np.isfinite(valid["interval_seconds"])
    & (valid["interval_seconds"] > 0)
    & np.isfinite(valid["distance_m"])
    & (valid["distance_m"] >= 0)
].copy()

valid["endpoint_speed_kmh"] = (
    valid["endpoint_path_length_m"]
    / valid["interval_seconds"]
    * 3.6
)

valid["endpoint_detour_ratio"] = (
    valid["endpoint_path_length_m"]
    / valid["distance_m"].clip(lower=10.0)
)

# Diagnostic flags only.
valid["flag_unreachable"] = ~valid["reachable"].astype(bool)
valid["flag_endpoint_speed_gt_130"] = (
    valid["endpoint_speed_kmh"] > 130
)
valid["flag_endpoint_detour_gt_5"] = (
    valid["endpoint_detour_ratio"] > 5
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
valid.drop(columns=["_merge"]).to_parquet(OUTPUT, index=False)

print("\n=== TRANSITION QUALITY ===")
print("Valid transitions:", len(valid))
print("Unreachable:", int(valid["flag_unreachable"].sum()))
print(
    "Endpoint speed > 130 km/h:",
    int(valid["flag_endpoint_speed_gt_130"].sum()),
)
print(
    "Endpoint detour ratio > 5:",
    int(valid["flag_endpoint_detour_gt_5"].sum()),
)

print("\nGPS interval duration (s):")
print(valid["interval_seconds"].describe().to_string())

print("\nGPS displacement (m):")
print(
    valid["distance_m"]
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .to_string()
)

print("\nEndpoint-based speed (km/h):")
print(
    valid["endpoint_speed_kmh"]
    .dropna()
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .to_string()
)

print("\nEndpoint detour ratio:")
print(
    valid["endpoint_detour_ratio"]
    .dropna()
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .to_string()
)

print("\nSaved:", OUTPUT)