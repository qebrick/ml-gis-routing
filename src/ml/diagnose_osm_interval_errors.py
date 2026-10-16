from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "data/processed/osm_interval_validation.parquet"
)

df = pd.read_parquet(INPUT_PATH).copy()

df["route_length_bin"] = pd.cut(
    df["route_distance_m"],
    bins=[0, 25, 50, 100, 200, 400, 1000, np.inf],
    labels=[
        "0-25",
        "25-50",
        "50-100",
        "100-200",
        "200-400",
        "400-1000",
        "1000+",
    ],
    right=False,
)

df["traversal_bin"] = pd.cut(
    df["traversal_records"],
    bins=[0, 1, 2, 3, 5, 10],
    labels=["1", "2", "3", "4-5", "6-9"],
    right=True,
    include_lowest=True,
)

print("=== ERROR BY ROUTE LENGTH ===")

by_length = (
    df.groupby("route_length_bin", observed=True)
    .agg(
        intervals=("trip_id", "size"),
        median_route_m=("route_distance_m", "median"),
        median_osm_time_s=("osm_time_s", "median"),
        mean_error_s=("time_error_s", "mean"),
        median_error_s=("time_error_s", "median"),
        mae_s=("absolute_time_error_s", "mean"),
    )
)

print(by_length.round(2).to_string())

print("\n=== ERROR BY TRAVERSAL COUNT ===")

by_count = (
    df.groupby("traversal_bin", observed=True)
    .agg(
        intervals=("trip_id", "size"),
        median_route_m=("route_distance_m", "median"),
        median_osm_time_s=("osm_time_s", "median"),
        mean_error_s=("time_error_s", "mean"),
        mae_s=("absolute_time_error_s", "mean"),
    )
)

print(by_count.round(2).to_string())

print("\n=== CORRELATIONS ===")

for column in [
    "route_distance_m",
    "traversal_records",
    "osm_time_s",
    "target_speed_kmh",
]:
    correlation = df[column].corr(df["time_error_s"])
    print(
        f"{column}: {correlation:.4f}"
    )

print("\n=== LARGE POSITIVE ERRORS ===")

print(
    df.nlargest(10, "time_error_s")[
        [
            "trip_id",
            "point_index",
            "route_distance_m",
            "traversal_records",
            "osm_time_s",
            "observed_time_s",
            "time_error_s",
        ]
    ]
    .round(2)
    .to_string(index=False)
)

print("\n=== LARGE NEGATIVE ERRORS ===")

print(
    df.nsmallest(10, "time_error_s")[
        [
            "trip_id",
            "point_index",
            "route_distance_m",
            "traversal_records",
            "osm_time_s",
            "observed_time_s",
            "time_error_s",
        ]
    ]
    .round(2)
    .to_string(index=False)
)
