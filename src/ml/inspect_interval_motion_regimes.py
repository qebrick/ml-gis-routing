from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "data/processed/osm_interval_validation.parquet"
)

df = pd.read_parquet(INPUT_PATH).copy()

df["osm_time_ratio"] = (
    df["osm_time_s"] / df["observed_time_s"]
)

df["motion_regime"] = pd.cut(
    df["target_speed_kmh"],
    bins=[0, 5, 15, 30, 50, 90, 130],
    labels=[
        "0-5",
        "5-15",
        "15-30",
        "30-50",
        "50-90",
        "90-130",
    ],
    right=False,
)

print("=== MOTION REGIMES ===")

summary = (
    df.groupby("motion_regime", observed=True)
    .agg(
        intervals=("trip_id", "size"),
        median_route_m=("route_distance_m", "median"),
        median_osm_time_s=("osm_time_s", "median"),
        mean_error_s=("time_error_s", "mean"),
        mae_s=("absolute_time_error_s", "mean"),
    )
)

print(summary.round(2).to_string())

print("\n=== OSM TIME RATIO ===")

print(
    df["osm_time_ratio"]
    .describe(
        percentiles=[
            0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99
        ]
    )
    .round(3)
    .to_string()
)

print("\n=== PLAUSIBILITY COUNTS ===")

conditions = {
    "speed_below_5": df["target_speed_kmh"] < 5,
    "speed_above_90": df["target_speed_kmh"] > 90,
    "osm_time_over_30s": df["osm_time_s"] > 30,
    "osm_time_over_45s": df["osm_time_s"] > 45,
    "osm_time_under_3s": df["osm_time_s"] < 3,
    "speed_5_to_90": df["target_speed_kmh"].between(
        5, 90, inclusive="both"
    ),
}

for name, mask in conditions.items():
    print(name, int(mask.sum()))

print("\n=== MODERATE-MOTION SUBSET ===")

moderate = df.loc[
    df["target_speed_kmh"].between(
        5, 90, inclusive="both"
    )
    & (df["osm_time_s"] <= 30)
].copy()

print("Intervals:", len(moderate))
print("Trips:", moderate["trip_id"].nunique())

if len(moderate):
    print(
        "OSM MAE (s):",
        round(moderate["absolute_time_error_s"].mean(), 3)
    )
    print(
        "OSM bias (s):",
        round(moderate["time_error_s"].mean(), 3)
    )

print("\n=== TOP OSM TIME RATIOS ===")

print(
    df.nlargest(10, "osm_time_ratio")[
        [
            "trip_id",
            "point_index",
            "route_distance_m",
            "target_speed_kmh",
            "osm_time_s",
            "osm_time_ratio",
        ]
    ]
    .round(2)
    .to_string(index=False)
)
