from pathlib import Path

import numpy as np
import pandas as pd

MATCHED = Path("data/processed/hmm_pilot.parquet")
INTERVALS = Path("data/processed/gps_intervals.parquet")
OUTPUT = Path("data/processed/hmm_interval_quality.parquet")

matched = pd.read_parquet(MATCHED).sort_values(
    ["hmm_fragment_id", "point_index"]
)

gps = pd.read_parquet(
    INTERVALS,
    columns=[
        "trip_id",
        "point_index",
        "distance_m",
        "interval_seconds",
    ],
)

# Each matched point except the first in a fragment
# describes the route from its preceding point.
matched["previous_index"] = (
    matched.groupby("hmm_fragment_id")["point_index"].shift(1)
)

transitions = matched[
    matched["previous_index"].notna()
].copy()

transitions["previous_index"] = (
    transitions["previous_index"].astype("int64")
)

if not (
    transitions["point_index"]
    == transitions["previous_index"] + 1
).all():
    raise RuntimeError("Non-consecutive HMM transition.")

transitions = transitions.merge(
    gps.rename(columns={
        "point_index": "previous_index",
        "distance_m": "gps_displacement_m",
    }),
    on=["trip_id", "previous_index"],
    how="left",
    validate="many_to_one",
)

if transitions["interval_seconds"].isna().any():
    raise RuntimeError("Missing GPS intervals.")

transitions["route_distance_m"] = (
    transitions["route_distance_from_previous_m"]
)

transitions["route_speed_kmh"] = (
    transitions["route_distance_m"]
    / transitions["interval_seconds"]
    * 3.6
)

transitions["gps_speed_kmh"] = (
    transitions["gps_displacement_m"]
    / transitions["interval_seconds"]
    * 3.6
)

transitions["distance_ratio"] = (
    transitions["route_distance_m"]
    / transitions["gps_displacement_m"].clip(lower=10)
)

transitions["flag_route_shorter_than_gps"] = (
    transitions["route_distance_m"] + 20
    < transitions["gps_displacement_m"]
)

transitions["flag_large_detour"] = (
    (transitions["distance_ratio"] > 3)
    & (transitions["gps_displacement_m"] >= 20)
)

transitions["flag_zero_route"] = (
    transitions["route_distance_m"] < 0.01
)

transitions["flag_zero_route_moving_gps"] = (
    transitions["flag_zero_route"]
    & (transitions["gps_displacement_m"] > 20)
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
transitions.to_parquet(OUTPUT, index=False)

print("=== HMM INTERVAL QUALITY ===")
print("Transitions:", len(transitions))
print(
    "Route shorter than GPS by >20m:",
    int(transitions["flag_route_shorter_than_gps"].sum()),
)
print(
    "Large detour (ratio >3, GPS >=20m):",
    int(transitions["flag_large_detour"].sum()),
)
print(
    "Zero-distance routes:",
    int(transitions["flag_zero_route"].sum()),
)
print(
    "Zero route with GPS displacement >20m:",
    int(transitions["flag_zero_route_moving_gps"].sum()),
)

print("\nRoute/GPS distance ratio:")
print(
    transitions["distance_ratio"]
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .to_string()
)

print("\nGPS speed (km/h):")
print(
    transitions["gps_speed_kmh"]
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .to_string()
)

print("\nRoute speed (km/h):")
print(
    transitions["route_speed_kmh"]
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .to_string()
)

print("\nSaved:", OUTPUT)
